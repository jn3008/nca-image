import { eraseShader, renderShader, updateShader } from "./shaders";

const CHANNELS = 16;
const PERCEPTION = 48;
const HIDDEN = 128;
const DEFAULT_SIM_PADDING = 0;

type ExportedModel = {
  format: "nca-webgpu-v1";
  step: number;
  modelConfig: {
    channels: number;
    hidden_size: number;
    fire_rate: number;
    perception: string;
  };
  grid: {
    width: number;
    height: number;
    channels: number;
  };
  weights: {
    w1Shape: number[];
    b1Shape: number[];
    w2Shape: number[];
    w1: number[];
    b1: number[];
    w2: number[];
  };
};

type GpuBundle = {
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
};

function getElement<T extends Element>(selector: string): T {
  const element = document.querySelector<T>(selector);
  if (!element) {
    throw new Error(`Missing required DOM element: ${selector}`);
  }
  return element;
}

const canvas = getElement<HTMLCanvasElement>("#canvas");
const statusEl = getElement<HTMLElement>("#status");
const toggleButton = getElement<HTMLButtonElement>("#toggle");
const stepButton = getElement<HTMLButtonElement>("#step");
const resetButton = getElement<HTMLButtonElement>("#reset");
const stepsInput = getElement<HTMLInputElement>("#stepsPerFrame");
const stepsValue = getElement<HTMLElement>("#stepsValue");
const paddingInput = getElement<HTMLInputElement>("#simPadding");
const paddingValue = getElement<HTMLElement>("#paddingValue");

function setStatus(message: string): void {
  statusEl.textContent = message;
}

async function initGpu(): Promise<GpuBundle> {
  if (!navigator.gpu) {
    throw new Error("WebGPU is not available in this browser. Try a current Chrome or Edge build.");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("No WebGPU adapter was found.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  if (!context) {
    throw new Error("Could not create a WebGPU canvas context.");
  }

  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque",
  });

  return { device, context, format };
}

function makeBuffer(
  device: GPUDevice,
  label: string,
  size: number,
  usage: GPUBufferUsageFlags,
  data?: ArrayBufferView,
): GPUBuffer {
  const buffer = device.createBuffer({
    label,
    size,
    usage,
    mappedAtCreation: data !== undefined,
  });

  if (data) {
    const mapped = buffer.getMappedRange();
    new Uint8Array(mapped).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    buffer.unmap();
  }

  return buffer;
}

function makeInitialState(width: number, height: number, seedColor = false): Float32Array {
  const state = new Float32Array(width * height * CHANNELS);
  const cx = Math.floor(width / 2);
  const cy = Math.floor(height / 2);
  const index = (cy * width + cx) * CHANNELS;
  for (let c = 3; c < CHANNELS; c += 1) {
    state[index + c] = 1.0;
  }
  if (seedColor) {
    state[index + 0] = 0.1;
    state[index + 1] = 0.65;
    state[index + 2] = 0.95;
  }
  return state;
}

function zeroWeights(): {
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
} {
  return {
    w1: new Float32Array(HIDDEN * PERCEPTION),
    b1: new Float32Array(HIDDEN),
    w2: new Float32Array(CHANNELS * HIDDEN),
  };
}

function writeBuffer(device: GPUDevice, buffer: GPUBuffer, data: ArrayBufferView): void {
  device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource);
}

let simWidth = 64;
let simHeight = 64;

function getSimulationPadding(): number {
  const parsed = Number.parseInt(paddingInput.value, 10);
  if (!Number.isFinite(parsed) || parsed < 0) {
    return DEFAULT_SIM_PADDING;
  }
  return Math.min(parsed, 64);
}

function writeParams(device: GPUDevice, buffer: GPUBuffer, tick: number, mode: number): void {
  const params = new Uint32Array([simWidth, simHeight, tick, mode]);
  writeBuffer(device, buffer, params);
}

function canvasToGrid(canvasX: number, canvasY: number): { x: number; y: number } {
  const rect = canvas.getBoundingClientRect();
  const x = Math.max(0, Math.min(simWidth - 1, Math.floor(((canvasX - rect.left) / rect.width) * simWidth)));
  const y = Math.max(0, Math.min(simHeight - 1, Math.floor(((canvasY - rect.top) / rect.height) * simHeight)));
  return { x, y };
}

async function loadExportedModel(): Promise<ExportedModel | null> {
  const response = await fetch("/model.json", { cache: "no-store" });
  if (!response.ok) {
    return null;
  }

  const model = (await response.json()) as ExportedModel;
  if (model.format !== "nca-webgpu-v1") {
    throw new Error(`Unsupported model format: ${model.format}`);
  }
  if (
    model.grid.channels !== CHANNELS ||
    model.modelConfig.channels !== CHANNELS ||
    model.modelConfig.hidden_size !== HIDDEN ||
    model.modelConfig.perception !== "sobel"
  ) {
    throw new Error("Exported model must use 16 channels, hidden_size 128, and sobel perception.");
  }
  return model;
}

function weightsFromModel(model: ExportedModel | null): {
  w1: Float32Array;
  b1: Float32Array;
  w2: Float32Array;
} {
  if (!model) {
    return zeroWeights();
  }

  const w1 = new Float32Array(model.weights.w1);
  const b1 = new Float32Array(model.weights.b1);
  const w2 = new Float32Array(model.weights.w2);
  if (w1.length !== HIDDEN * PERCEPTION || b1.length !== HIDDEN || w2.length !== CHANNELS * HIDDEN) {
    throw new Error("Exported model weight sizes do not match the WebGPU simulator.");
  }
  return { w1, b1, w2 };
}

async function main(): Promise<void> {
  const { device, context, format } = await initGpu();
  const exportedModel = await loadExportedModel();
  let simPadding = exportedModel ? getSimulationPadding() : 0;
  if (exportedModel) {
    simWidth = exportedModel.grid.width + simPadding * 2;
    simHeight = exportedModel.grid.height + simPadding * 2;
  }

  const weightUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const stateUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const uniformUsage = GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST;

  let cpuState = makeInitialState(simWidth, simHeight, exportedModel === null);
  const weights = weightsFromModel(exportedModel);
  const mode = exportedModel ? 1 : 0;

  function makeStateBuffer(label: string, data?: Float32Array): GPUBuffer {
    return makeBuffer(
      device,
      label,
      simWidth * simHeight * CHANNELS * Float32Array.BYTES_PER_ELEMENT,
      stateUsage,
      data,
    );
  }

  let stateA = makeStateBuffer("state A", cpuState);
  let stateB = makeStateBuffer("state B");
  const w1 = makeBuffer(device, "w1", weights.w1.byteLength, weightUsage, weights.w1);
  const b1 = makeBuffer(device, "b1", weights.b1.byteLength, weightUsage, weights.b1);
  const w2 = makeBuffer(device, "w2", weights.w2.byteLength, weightUsage, weights.w2);
  const paramsBuffer = makeBuffer(device, "params", 16, uniformUsage);
  const eraseParamsBuffer = makeBuffer(device, "erase params", 32, uniformUsage);

  const updateModule = device.createShaderModule({ label: "NCA update", code: updateShader });
  const renderModule = device.createShaderModule({ label: "NCA render", code: renderShader });
  const eraseModule = device.createShaderModule({ label: "NCA erase", code: eraseShader });

  const updatePipeline = device.createComputePipeline({
    label: "NCA update pipeline",
    layout: "auto",
    compute: { module: updateModule, entryPoint: "main" },
  });

  const renderPipeline = device.createRenderPipeline({
    label: "NCA render pipeline",
    layout: "auto",
    vertex: { module: renderModule, entryPoint: "vs" },
    fragment: {
      module: renderModule,
      entryPoint: "fs",
      targets: [{ format }],
    },
    primitive: { topology: "triangle-list" },
  });

  const erasePipeline = device.createComputePipeline({
    label: "NCA erase pipeline",
    layout: "auto",
    compute: { module: eraseModule, entryPoint: "main" },
  });

  function makeUpdateBindGroups(): GPUBindGroup[] {
    return [
      device.createBindGroup({
        label: "update A to B",
        layout: updatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateA } },
          { binding: 1, resource: { buffer: stateB } },
          { binding: 2, resource: { buffer: w1 } },
          { binding: 3, resource: { buffer: b1 } },
          { binding: 4, resource: { buffer: w2 } },
          { binding: 5, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        label: "update B to A",
        layout: updatePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateB } },
          { binding: 1, resource: { buffer: stateA } },
          { binding: 2, resource: { buffer: w1 } },
          { binding: 3, resource: { buffer: b1 } },
          { binding: 4, resource: { buffer: w2 } },
          { binding: 5, resource: { buffer: paramsBuffer } },
        ],
      }),
    ];
  }

  function makeRenderBindGroups(): GPUBindGroup[] {
    return [
      device.createBindGroup({
        label: "render A",
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateA } },
          { binding: 1, resource: { buffer: paramsBuffer } },
        ],
      }),
      device.createBindGroup({
        label: "render B",
        layout: renderPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateB } },
          { binding: 1, resource: { buffer: paramsBuffer } },
        ],
      }),
    ];
  }

  function makeEraseBindGroups(): GPUBindGroup[] {
    return [
      device.createBindGroup({
        label: "erase A",
        layout: erasePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateA } },
          { binding: 1, resource: { buffer: eraseParamsBuffer } },
        ],
      }),
      device.createBindGroup({
        label: "erase B",
        layout: erasePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: stateB } },
          { binding: 1, resource: { buffer: eraseParamsBuffer } },
        ],
      }),
    ];
  }

  let updateBindGroups = makeUpdateBindGroups();
  let renderBindGroups = makeRenderBindGroups();
  let eraseBindGroups = makeEraseBindGroups();

  let readIndex = 0;
  let tick = 0;
  let running = true;
  let pointerDown = false;

  function reset(): void {
    cpuState = makeInitialState(simWidth, simHeight, exportedModel === null);
    writeBuffer(device, stateA, cpuState);
    writeBuffer(device, stateB, cpuState);
    readIndex = 0;
    tick = 0;
  }

  function updateModelStatus(): void {
    if (!exportedModel) {
      return;
    }
    setStatus(
      `Loaded model.json: trained ${exportedModel.grid.width}x${exportedModel.grid.height}, sim ${simWidth}x${simHeight}, step ${exportedModel.step}`,
    );
  }

  function setGridPadding(padding: number): void {
    if (!exportedModel) {
      return;
    }

    simPadding = padding;
    simWidth = exportedModel.grid.width + simPadding * 2;
    simHeight = exportedModel.grid.height + simPadding * 2;
    cpuState = makeInitialState(simWidth, simHeight, false);

    stateA.destroy();
    stateB.destroy();
    stateA = makeStateBuffer("state A", cpuState);
    stateB = makeStateBuffer("state B", cpuState);
    updateBindGroups = makeUpdateBindGroups();
    renderBindGroups = makeRenderBindGroups();
    eraseBindGroups = makeEraseBindGroups();
    readIndex = 0;
    tick = 0;
    paddingValue.textContent = String(simPadding);
    updateModelStatus();
  }

  function encodeComputeStep(encoder: GPUCommandEncoder): void {
    writeParams(device, paramsBuffer, tick, mode);
    const pass = encoder.beginComputePass();
    pass.setPipeline(updatePipeline);
    pass.setBindGroup(0, updateBindGroups[readIndex]);
    pass.dispatchWorkgroups(Math.ceil(simWidth / 8), Math.ceil(simHeight / 8));
    pass.end();
    readIndex = 1 - readIndex;
    tick += 1;
  }

  function computeStep(): void {
    const encoder = device.createCommandEncoder();
    encodeComputeStep(encoder);
    device.queue.submit([encoder.finish()]);
  }

  function eraseAt(canvasX: number, canvasY: number): void {
    const position = canvasToGrid(canvasX, canvasY);
    const params = new Uint32Array([simWidth, simHeight, position.x, position.y, 3, 0, 0, 0]);
    writeBuffer(device, eraseParamsBuffer, params);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(erasePipeline);
    pass.setBindGroup(0, eraseBindGroups[readIndex]);
    pass.dispatchWorkgroups(Math.ceil(simWidth / 8), Math.ceil(simHeight / 8));
    pass.end();
    device.queue.submit([encoder.finish()]);
  }

  function render(encoder: GPUCommandEncoder): void {
    writeParams(device, paramsBuffer, tick, mode);
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.07, g: 0.085, b: 0.095, a: 1 },
          loadOp: "clear",
          storeOp: "store",
        },
      ],
    });
    pass.setPipeline(renderPipeline);
    pass.setBindGroup(0, renderBindGroups[readIndex]);
    pass.draw(3);
    pass.end();
  }

  function frame(): void {
    const encoder = device.createCommandEncoder();
    if (running) {
      const steps = Number(stepsInput.value);
      for (let i = 0; i < steps; i += 1) {
        computeStep();
      }
    }
    render(encoder);
    device.queue.submit([encoder.finish()]);
    requestAnimationFrame(frame);
  }

  toggleButton.addEventListener("click", () => {
    running = !running;
    toggleButton.textContent = running ? "Pause" : "Run";
  });

  stepButton.addEventListener("click", () => {
    computeStep();
    const encoder = device.createCommandEncoder();
    render(encoder);
    device.queue.submit([encoder.finish()]);
  });

  resetButton.addEventListener("click", reset);

  stepsInput.addEventListener("input", () => {
    stepsValue.textContent = stepsInput.value;
  });

  paddingInput.addEventListener("input", () => {
    setGridPadding(getSimulationPadding());
  });

  canvas.addEventListener("pointerdown", (event) => {
    pointerDown = true;
    eraseAt(event.clientX, event.clientY);
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!pointerDown) {
      return;
    }
    eraseAt(event.clientX, event.clientY);
  });

  window.addEventListener("pointerup", () => {
    pointerDown = false;
  });

  if (exportedModel) {
    paddingInput.value = String(simPadding);
    paddingValue.textContent = String(simPadding);
    updateModelStatus();
  } else {
    paddingInput.disabled = true;
    setStatus("No /model.json found. Running built-in demo mode with the 16-channel state layout.");
  }
  frame();
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  setStatus(message);
  console.error(error);
});
