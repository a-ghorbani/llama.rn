# llama.rn example

This is an example project to show how to use the llama.rn library.

For iPhone/iPad/Mac, you can try it by downloading our test app from [TestFlight](https://testflight.apple.com/join/MmzGSneU).

## Examples

The example app demonstrates various local LLM capabilities:

- **💬 Simple Chat** - Basic chat interface with text generation ([SimpleChatScreen.tsx](src/screens/SimpleChatScreen.tsx))
- **👁️ Vision/Multimodal** - Image analysis and visual question answering ([MultimodalScreen.tsx](src/screens/MultimodalScreen.tsx))
- **🛠️ Tool Calling** - Advanced function calling capabilities ([ToolCallsScreen.tsx](src/screens/ToolCallsScreen.tsx))
- **🔊 Text-to-Speech** - Local voice synthesis with OuteTTS ([TTSScreen.tsx](src/screens/TTSScreen.tsx))
- **📊 Model Info** - Model diagnostics and system information ([ModelInfoScreen.tsx](src/screens/ModelInfoScreen.tsx))

Used models are listed in [src/utils/constants.ts](src/utils/constants.ts).

## Requirements

Please back to the root directory and run the following command:

```bash
npm install && npm run bootstrap
```

## iOS

1. Install pods

```bash
npm run pods
```

2. Run the example

```bash
npm run ios
# Use device
npm run ios -- --device "<device name>"
# With release mode
npm run ios -- --mode Release
```

## Android

Run the example:
```bash
npm run android
# With release mode
npm run android -- --mode release
```

## Build with frameworks/libs

This example is build llama.rn from source code by default, you can also build with frameworks/libs.

```bash
# Build iOS frameworks
npm run build:ios-frameworks
# Build Android libs
npm run build:android-libs
```

Then you can setup the environment variable / properties in your project:

iOS:
```bash
RNLLAMA_BUILD_FROM_SOURCE=0 npm run pods
```

Android: Edit `android/gradle.properties` and set `rnllamaBuildFromSource` to `false`.
