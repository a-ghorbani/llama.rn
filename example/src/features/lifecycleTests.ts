/* eslint-disable no-await-in-loop */
import { initLlama, LlamaContext } from '../../../src'

// Stop and release while the context is busy with a long prompt. A single
// llama_decode over the whole prompt is the worst case: stop and release
// can only act on it from inside the decode, through the abort callback.

export type LifecycleLog = (message: string) => void

export type LifecycleOptions = {
  modelPath: string
  threads: number
  log: LifecycleLog
}

const PROMPT_TOKENS = 1800
const N_CTX = 4096
// One batch for the whole prompt, so the prefill is one long llama_decode
const N_BATCH = 2048
// Long enough for the prefill to be underway on any device
const PREFILL_HEAD_START_MS = 2000

const PARAGRAPH =
  'The river town woke slowly. Fishermen checked their nets, bakers lit ' +
  'their ovens, and the ferry captain counted the crates stacked on the ' +
  'pier while gulls argued over scraps near the market stalls. '

const delay = (ms: number) =>
  new Promise<void>((resolve) => {
    setTimeout(resolve, ms)
  })

const elapsed = (start: number) => `${Date.now() - start} ms`

const initCpuContext = (modelPath: string, threads: number) =>
  initLlama({
    model: modelPath,
    n_ctx: N_CTX,
    n_batch: N_BATCH,
    n_threads: threads,
    n_gpu_layers: 0,
    no_gpu_devices: true,
  })

const buildLongPrompt = async (ctx: LlamaContext) => {
  let prompt = ''
  let nTokens = 0
  while (nTokens < PROMPT_TOKENS) {
    prompt += PARAGRAPH.repeat(10)
    nTokens = (await ctx.tokenize(prompt)).tokens.length
  }
  return {
    prompt: `${prompt}\nSummarize the text above in one sentence.`,
    nTokens,
  }
}

const describeResult = (result: unknown) => {
  if (!result || typeof result !== 'object') return String(result)
  const r = result as Record<string, unknown>
  return `interrupted=${r.interrupted} tokens_predicted=${r.tokens_predicted}${
    r.context_released ? ' context_released=true' : ''
  }`
}

const prepare = async ({ modelPath, threads, log }: LifecycleOptions) => {
  log(`init: CPU, n_threads=${threads}, n_batch=${N_BATCH}`)
  const ctx = await initCpuContext(modelPath, threads)
  const { prompt, nTokens } = await buildLongPrompt(ctx)
  log(`prompt: ${nTokens} tokens`)
  return { ctx, prompt }
}

// Release mid-prefill. Before contexts were reference counted, release gave
// the worker 5 s and then destroyed the context under it.
export async function releaseDuringPrefill(options: LifecycleOptions) {
  const { log } = options
  const { ctx, prompt } = await prepare(options)

  const start = Date.now()
  const completion = ctx
    .completion({ prompt, n_predict: 32 })
    .then((result) => `resolved: ${describeResult(result)}`)
    .catch((error: Error) => `rejected: ${error.message}`)

  await delay(PREFILL_HEAD_START_MS)
  log(`release: start (${elapsed(start)} into the completion)`)
  const releaseStart = Date.now()
  await ctx.release()
  log(`release: done in ${elapsed(releaseStart)}`)
  log(`completion ${await completion}`)
}

// Stop mid-prefill: how long until the completion promise settles.
export async function stopDuringPrefill(options: LifecycleOptions) {
  const { log } = options
  const { ctx, prompt } = await prepare(options)
  try {
    const start = Date.now()
    const completion = ctx.completion({ prompt, n_predict: 32 })

    await delay(PREFILL_HEAD_START_MS)
    log(`stop: start (${elapsed(start)} into the completion)`)
    const stopStart = Date.now()
    await ctx.stopCompletion()
    const result = await completion
    log(`stop: completion settled ${elapsed(stopStart)} after stop`)
    log(`completion resolved: ${describeResult(result)}`)
  } finally {
    await ctx.release()
  }
}

// Stop right after completion() is called, while the chat prompt is still
// being formatted.
export async function stopDuringFormatting(options: LifecycleOptions) {
  const { modelPath, threads, log } = options
  const nPredict = 128
  log(`init: CPU, n_threads=${threads}`)
  const ctx = await initCpuContext(modelPath, threads)
  try {
    const start = Date.now()
    const completion = ctx.completion({
      messages: [{ role: 'user', content: 'Count from 1 to 1000.' }],
      n_predict: nPredict,
      ignore_eos: true,
    })
    await ctx.stopCompletion()
    log('stop: called right after completion()')
    const result = await completion
    log(`completion resolved in ${elapsed(start)}: ${describeResult(result)}`)
    log(
      result.interrupted
        ? 'stop took effect'
        : `stop was lost: generated up to n_predict=${nPredict}`,
    )
  } finally {
    await ctx.release()
  }
}
