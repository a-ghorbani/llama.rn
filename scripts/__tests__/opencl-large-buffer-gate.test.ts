// ggml-opencl adds -qcom-enable-large-buffer to the kernel compile options
// when adreno_use_large_buffer is set. A driver that rejects the option
// (Adreno 630: err -43) fails every kernel build, and a failed build is fatal.
// So the flag must be final, and checked against the driver, before
// ggml_cl_init builds kernel_compile_opts; upstream turned it off later, in
// ggml_opencl_print_backend_info, after the kernels were built.
import * as fs from 'fs'
import * as path from 'path'

const SOURCE = path.join(
  __dirname,
  '..',
  '..',
  'vendor/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp',
)
const INIT_HEAD =
  'static ggml_backend_opencl_context * ggml_cl_init(ggml_backend_dev_t dev) {'

const largeBufferProblems = (src: string): string[] => {
  const problems: string[] = []
  const init = src.indexOf(INIT_HEAD)
  if (init < 0) return ['ggml_cl_init definition not found']
  const opts = src.indexOf('backend_ctx->kernel_compile_opts =', init)
  if (opts < 0) return ['kernel_compile_opts assignment not found in ggml_cl_init']
  // plain `=` only; a default member initializer upstream would be flagged too
  const writes = [...src.matchAll(/adreno_use_large_buffer\s*=(?!=)/g)].map(
    (m) => m.index ?? 0,
  )
  if (writes.length === 0) problems.push('adreno_use_large_buffer is never set')
  writes
    .filter((at) => at < init || at > opts)
    .forEach((at) => {
      const line = src.slice(0, at).split('\n').length
      problems.push(
        `adreno_use_large_buffer set at line ${line}, outside ggml_cl_init before kernel_compile_opts`,
      )
    })
  const decision = src.slice(writes[0] ?? opts, opts)
  if (!decision.includes('adreno_has_large_buffer'))
    problems.push('large buffer not gated on cl_qcom_large_buffer')
  if (!decision.includes('ggml_opencl_compiler_accepts('))
    problems.push('large buffer not checked against the compiler')
  return problems
}

describe('ggml-opencl large buffer gate', () => {
  it('decides large buffer before building the compile options', () => {
    expect(largeBufferProblems(fs.readFileSync(SOURCE, 'utf8'))).toEqual([])
  })

  it('flags the upstream late check', () => {
    const src = [
      INIT_HEAD,
      '  backend_ctx->adreno_use_large_buffer = getenv("X") != nullptr;',
      '  backend_ctx->kernel_compile_opts = make(backend_ctx);',
      '}',
      'static void print_info() {',
      '  if (!backend_ctx->adreno_has_large_buffer) backend_ctx->adreno_use_large_buffer = false;',
      '}',
    ].join('\n')
    expect(largeBufferProblems(src)).toEqual([
      'adreno_use_large_buffer set at line 6, outside ggml_cl_init before kernel_compile_opts',
      'large buffer not gated on cl_qcom_large_buffer',
      'large buffer not checked against the compiler',
    ])
  })
})
