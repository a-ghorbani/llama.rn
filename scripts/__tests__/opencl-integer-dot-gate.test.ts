// ggml-opencl builds every program in load_cl_kernels at GPU init, and a
// failed build is fatal (exit(1)). A program that calls an integer-dot builtin
// fails to build on drivers without cl_khr_integer_dot_product (Adreno 630,
// Adreno 730 on Snapdragon 8 Gen 1), so its load site must check
// backend_ctx->has_integer_dot. llama.cpp syncs keep adding such programs.
// A gate through a local variable (`bool ok = ...; if (ok)`) is flagged too:
// a false alarm at sync time is cheaper than a missed gate.
import * as fs from 'fs'
import * as path from 'path'

const OPENCL_DIR = path.join(
  __dirname,
  '..',
  '..',
  'vendor/llama.cpp/ggml/src/ggml-opencl',
)
const GATE = 'backend_ctx->has_integer_dot'
const DOT_BUILTIN = /\bdot(?:_acc_sat)?_4x8packed_\w+\s*\(/
const FUNCTION_HEAD =
  /^(?!(?:if|for|while|switch|else)\b)[\s\w&*:<>]+\b\w+\s*\([^()]*\)\s*(?:const\s*)?$/

// Blank out comments and string contents, keeping offsets and newlines.
const stripComments = (src: string): string =>
  src.replace(
    /\/\/[^\n]*|\/\*[\S\s]*?\*\/|"(?:\\.|[^\n"\\])*"|'(?:\\.|[^\n'\\])*'/g,
    (m) =>
      m.startsWith('/')
        ? m.replace(/[^\n]/g, ' ')
        : m[0] + ' '.repeat(m.length - 2) + m[0],
  )

const splitTop = (expr: string, op: string): string[] => {
  const parts: string[] = []
  let depth = 0
  let start = 0
  let i = 0
  while (i < expr.length) {
    if (expr[i] === '(') depth += 1
    if (expr[i] === ')') depth -= 1
    if (depth === 0 && expr.startsWith(op, i)) {
      parts.push(expr.slice(start, i).trim())
      start = i + op.length
      i += op.length
    } else {
      i += 1
    }
  }
  parts.push(expr.slice(start).trim())
  return parts
}

const unwrap = (expr: string): string => {
  const e = expr.trim()
  if (!e.startsWith('(') || !e.endsWith(')')) return e
  const inner = e.slice(1, -1)
  // `(a) && (b)` starts and ends with parens but is not wrapped
  const balanced = [...inner].reduce(
    (depth, c) => {
      if (depth < 0) return depth
      if (c === '(') return depth + 1
      return c === ')' ? depth - 1 : depth
    },
    0,
  )
  return balanced === 0 ? unwrap(inner) : e
}

// Whether a conjunction (no top-level ||) has one term matching `term`.
const requiresTerm = (cond: string, term: (t: string) => boolean): boolean => {
  const e = unwrap(cond)
  return (
    splitTop(e, '||').length === 1 &&
    splitTop(e, '&&').some((t) => term(unwrap(t)))
  )
}

/**
 * Whether the program source calls an integer-dot builtin outside an
 * #if/#ifdef on cl_khr_integer_dot_product (or a macro defined only under it).
 */
function usesIntegerDotUnguarded(clSource: string): boolean {
  const guards = new Set(['cl_khr_integer_dot_product'])
  const isGuard = (t: string) => {
    const m = t.match(/^(?:defined\s*\(\s*(\w+)\s*\)|defined\s+(\w+)|(\w+))$/)
    return !!m && guards.has(m[1] ?? m[2] ?? m[3] ?? '')
  }
  const frames: boolean[] = []
  return stripComments(clSource)
    .split('\n')
    .some((raw) => {
      const line = raw.trim()
      const [, name = '', rest = ''] = line.match(/^#\s*(\w+)\s*(.*)$/) ?? []
      if (name === 'ifdef') frames.push(guards.has(rest.trim()))
      else if (name === 'ifndef') frames.push(false)
      else if (name === 'if') frames.push(requiresTerm(rest, isGuard))
      else if (name === 'elif' || name === 'else')
        frames.splice(-1, 1, false)
      else if (name === 'endif') frames.pop()
      else if (name === 'define' && frames.some(Boolean))
        guards.add(rest.trim().split(/[\s(]/)[0] ?? '')
      return DOT_BUILTIN.test(line) && !frames.some(Boolean)
    })
}

interface Opener {
  head: string
  bodyStart: number
}

// The brace blocks open at `offset`, outermost first, with their heads.
const openersAt = (code: string, offset: number): Opener[] => {
  const stack: Opener[] = []
  let headStart = 0
  ;[...code.slice(0, offset)].forEach((c, i) => {
    if (c === '{') {
      const head = code
        .slice(headStart, i)
        .replace(/^\s*#.*$/gm, '')
        .replace(/\s+/g, ' ')
        .trim()
      stack.push({ head, bodyStart: i + 1 })
    }
    if (c === '}') stack.pop()
    if (c === '{' || c === '}' || c === ';') headStart = i + 1
  })
  return stack
}

const bodyOf = (code: string, bodyStart: number): string => {
  let depth = 1
  let i = bodyStart
  while (i < code.length && depth > 0) {
    if (code[i] === '{') depth += 1
    if (code[i] === '}') depth -= 1
    i += 1
  }
  return code.slice(bodyStart, depth === 0 ? i - 1 : i)
}

const NOT_GATE = /^!\s*backend_ctx->has_integer_dot$/

// `static bool pred(backend_ctx)` that returns false without integer dot
// before it can return true.
const predicateRequiresGate = (code: string, name: string): boolean => {
  const def = new RegExp(`\\bstatic\\s+bool\\s+${name}\\s*\\([^)]*\\)\\s*\\{`)
  const m = def.exec(code)
  if (!m) return false
  const body = bodyOf(code, m.index + m[0].length)
  const refused = body.search(
    /if\s*\(\s*!\s*backend_ctx->has_integer_dot\s*\)\s*{?\s*return\s+false\s*;/,
  )
  const firstTrue = body.search(/return\s+true\s*;/)
  return refused !== -1 && (firstTrue === -1 || refused < firstTrue)
}

// The function starts with `if (... || !has_integer_dot || ...) return;`.
const returnsUnlessIntegerDot = (code: string, fn: Opener): boolean => {
  const first = bodyOf(code, fn.bodyStart).match(
    /^\s*if\s*\(([^;{}]*)\)\s*{?\s*return\s*;/,
  )
  if (!first?.[1]) return false
  return splitTop(unwrap(first[1]), '||').some((part) => {
    const p = unwrap(part)
    const call = p.match(/^!\s*(\w+)\s*\(\s*backend_ctx\s*\)$/)
    return NOT_GATE.test(p) || (!!call?.[1] && predicateRequiresGate(code, call[1]))
  })
}

/**
 * Programs whose load site in ggml-opencl.cpp can run on a device without
 * cl_khr_integer_dot_product.
 */
function ungatedIntegerDotLoads(
  cppSource: string,
  programs: string[],
): string[] {
  const code = stripComments(cppSource)
  return programs.filter((program) =>
    [...cppSource.matchAll(new RegExp(`#include\\s+"${program.replace(/\./g, '\\.')}\\.h"`, 'g'))]
      .map((match) => openersAt(code, match.index ?? 0))
      .some((openers) => {
        const gatedBlock = openers.some((o) => {
          const m = o.head.match(/^(?:else\s+)?if\s*\(([\S\s]*)\)$/)
          return !!m?.[1] && requiresTerm(m[1], (t) => t === GATE)
        })
        const fn = openers.find((o) => FUNCTION_HEAD.test(o.head))
        return !gatedBlock && !(fn && returnsUnlessIntegerDot(code, fn))
      }),
  )
}

const kernelsDir = path.join(OPENCL_DIR, 'kernels')
const programs = fs
  .readdirSync(kernelsDir)
  .filter((name) => name.endsWith('.cl'))
  .filter((name) =>
    usesIntegerDotUnguarded(fs.readFileSync(path.join(kernelsDir, name), 'utf-8')),
  )
  .sort()
const openclCpp = fs.readFileSync(
  path.join(OPENCL_DIR, 'ggml-opencl.cpp'),
  'utf-8',
)

describe('vendored ggml-opencl', () => {
  it('finds the integer-dot programs and a load site for each', () => {
    expect(programs).toContain('gemm_noshuffle_q2_k_q8_1_dp4a.cl')
    expect(programs).toContain('mul_mm_q8_f32_kqv.cl')
    expect(programs).not.toContain('flash_attn_f32_q8_0.cl')
    expect(
      programs.filter((p) => !openclCpp.includes(`#include "${p}.h"`)),
    ).toEqual([])
  })

  it('balances the braces of ggml-opencl.cpp', () => {
    const code = stripComments(openclCpp)
    expect(openersAt(code, code.length)).toEqual([])
  })

  it('loads integer-dot programs only when the device has integer dot', () => {
    expect(ungatedIntegerDotLoads(openclCpp, programs)).toEqual([])
  })
})

const DOT_CALL = 'r = dot_acc_sat_4x8packed_ss_int(a, b, r);'

describe('usesIntegerDotUnguarded', () => {
  it.each([
    ['a bare call', DOT_CALL],
    [
      'a call after a pragma-only #ifdef',
      `#ifdef cl_khr_integer_dot_product\n#pragma OPENCL EXTENSION cl_khr_integer_dot_product : enable\n#endif\n${DOT_CALL}`,
    ],
    [
      'a call in the #else branch',
      `#ifdef cl_khr_integer_dot_product\nr = 0;\n#else\n${DOT_CALL}\n#endif`,
    ],
    [
      'a call under an #if with ||',
      `#if defined(cl_khr_integer_dot_product) || defined(FOO)\n${DOT_CALL}\n#endif`,
    ],
  ])('flags %s', (_name, src) => {
    expect(usesIntegerDotUnguarded(src)).toBe(true)
  })

  it.each([
    ['a call in a comment', `// ${DOT_CALL}\n/* ${DOT_CALL} */`],
    [
      'a call under #if defined(...) && ...',
      `#if defined(cl_khr_integer_dot_product) && defined(FOO)\n${DOT_CALL}\n#endif`,
    ],
    [
      'a call under a macro defined only with integer dot',
      `#ifdef cl_khr_integer_dot_product\n#define HAVE_DOT 1\n#endif\n#ifdef HAVE_DOT\n${DOT_CALL}\n#endif`,
    ],
  ])('passes %s', (_name, src) => {
    expect(usesIntegerDotUnguarded(src)).toBe(false)
  })
})

const P = 'gemm_x_dp4a.cl'
const load = (opener: string) => `
static void load_cl_kernels(ggml_backend_opencl_context *backend_ctx) {
    // ${P}
    ${opener}
        const std::string kernel_src {
            #include "${P}.h"
        };
        GGML_LOG_CONT(".");
    }
}
`
const loader = (cond: string, predicate = '') => `${predicate}
static void load_x(ggml_backend_opencl_context * backend_ctx) {
    if (${cond}) {
        return;
    }
    {
        const std::string kernel_src {
            #include "${P}.h"
        };
    }
}`
const predicate = (body: string) =>
  `static bool x_on(const ggml_backend_opencl_context * backend_ctx) {\n${body}\n}`
const REFUSE = 'if (!backend_ctx->has_integer_dot) {\n    return false;\n}'

describe('ungatedIntegerDotLoads', () => {
  it.each([
    ['a bare block', load('{')],
    ['a block gated on something else', load('if (backend_ctx->has_subgroup_shuffle) {')],
    ['a block gated with ||', load(`if (${GATE} || force) {`)],
    ['a block gated on the negation', load(`if (!${GATE}) {`)],
    ['an early return on something else', loader('backend_ctx->gpu_family != ADRENO')],
    [
      'a gated early return after the load',
      `static void load_x(ggml_backend_opencl_context * backend_ctx) {
    if (backend_ctx->gpu_family == ADRENO) {
        const std::string kernel_src {
            #include "${P}.h"
        };
    }
    if (backend_ctx->gpu_family != ADRENO || !${GATE}) {
        return;
    }
}`,
    ],
    [
      'a predicate that can say yes first',
      loader('!x_on(backend_ctx)', predicate(`if (getenv("X")) {\n    return true;\n}\n${REFUSE}\nreturn true;`)),
    ],
  ])('flags %s', (_name, cpp) => {
    expect(ungatedIntegerDotLoads(cpp, [P])).toEqual([P])
  })

  it.each([
    ['a gated block', load(`if (${GATE}) {`)],
    ['a gated block after a long comment', `// ${'{'.repeat(40)}\n${load(`if (${GATE}) {`)}`],
    ['a block gated with &&', load(`if (${GATE} && backend_ctx->gpu_family == ADRENO) {`)],
    [
      'a gated block after a preprocessor line',
      load(`#ifdef GGML_OPENCL_USE_ADRENO_KERNELS\n    if (${GATE}) {`),
    ],
    ['an early return without integer dot', loader(`backend_ctx->gpu_family != ADRENO || !${GATE}`)],
    [
      'an early return through a predicate requiring integer dot',
      loader('!x_on(backend_ctx)', predicate(`${REFUSE}\nreturn true;`)),
    ],
  ])('passes %s', (_name, cpp) => {
    expect(ungatedIntegerDotLoads(cpp, [P])).toEqual([])
  })
})
