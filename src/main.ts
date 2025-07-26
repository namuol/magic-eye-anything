import {
  DepthEstimationPipelineOutput,
  pipeline,
  RawImage,
} from '@huggingface/transformers';
import GUI from 'lil-gui';

import {PixelGrid} from './PixelGrid';

// Available patterns in the public folder
const PRESET_PATTERNS = [
  {name: 'Carpet', url: 'carpet.jpg'},
  {name: 'Circuit', url: 'circuit.png'},
  {name: '90s', url: '90s.png'},
  {name: 'Marbles', url: 'marbles.jpg'},
  {name: 'Flowers', url: 'flowers.jpg'},
  {name: 'Flowers (vintage)', url: 'vintage-flowers-sm.jpg'},
] as const;

// Global state for autostereogram generation
const appState = {
  disparityScale: 1,
  selectedPattern: 'flowers.jpg',
  customPatternFile: null as File | null,
  currentImage: null as RawImage | null,
  currentDepth: null as PixelGrid | null,
  isProcessing: false,
  gui: null as GUI | null,
};

/**
 * Generates an autostereogram from the current depth image and pattern
 */
async function generateAutostereogram(): Promise<void> {
  if (!appState.currentDepth || appState.isProcessing) {
    return;
  }

  appState.isProcessing = true;
  show('loading-depth-estimation');
  hide('canvas');

  try {
    const canvasElement = document.getElementById(
      'canvas',
    ) as HTMLCanvasElement;
    const hiddenImageCanvas = new OffscreenCanvas(
      canvasElement.width,
      canvasElement.height,
    );

    const minDisparity = Math.floor(hiddenImageCanvas.width * 0.15);
    const maxDisparity = Math.floor(hiddenImageCanvas.width * 0.2);
    const tileWidth = minDisparity;

    // Load the pattern image
    let patternImageUrl: string;
    if (appState.customPatternFile) {
      patternImageUrl = URL.createObjectURL(appState.customPatternFile);
    } else {
      patternImageUrl = appState.selectedPattern;
    }

    const patternImage = (await RawImage.fromURL(patternImageUrl)).toCanvas();
    const tileHeight = (tileWidth * patternImage.height) / patternImage.width;
    const patternCanvas = new OffscreenCanvas(tileWidth, tileHeight);
    fillImage(patternImage, patternCanvas, tileWidth);

    const pattern = new PixelGrid(
      patternCanvas
        .getContext('2d')!
        .getImageData(0, 0, patternCanvas.width, patternCanvas.height),
    );

    const outputCanvas = new OffscreenCanvas(
      hiddenImageCanvas.width,
      hiddenImageCanvas.height,
    );
    const output = new PixelGrid(
      outputCanvas
        .getContext('2d')!
        .getImageData(0, 0, outputCanvas.width, outputCanvas.height),
    );

    // Generate the autostereogram
    for (let y = 0; y < output.height; ++y) {
      for (let x = 0; x < output.width; ++x) {
        const disparity = appState.currentDepth.get(x, y)[0] / 255;
        const offset = Math.floor(
          disparity * (maxDisparity - minDisparity) * appState.disparityScale,
        );
        if (x < minDisparity) {
          output.set(x, y, pattern.get((x + offset) % minDisparity, y));
        } else {
          output.set(x, y, output.get(x + offset - minDisparity, y));
        }
      }
    }

    const canvas = document.getElementById('canvas') as HTMLCanvasElement;
    const ctx = canvas.getContext('2d')!;
    ctx.putImageData(output.imageData, 0, 0);

    // Clean up object URL if we created one
    if (appState.customPatternFile) {
      URL.revokeObjectURL(patternImageUrl);
    }

    hide('loading-depth-estimation');
    show('canvas');
    show('save-image-container');
  } catch (error) {
    console.error('Error generating autostereogram:', error);
    hide('loading-depth-estimation');
  } finally {
    appState.isProcessing = false;
  }
}

/**
 * Sets up the GUI controls
 */
function setupGUI(): void {
  const gui = new GUI();
  gui.title('Controls');

  // Hide the GUI initially
  gui.hide();

  // Store GUI reference in app state
  appState.gui = gui;

  // Disparity scale slider
  gui
    .add(appState, 'disparityScale', 0.1, 1.75, 0.01)
    .name('Disparity')
    .onChange(
      debounce(() => {
        generateAutostereogram();
      }, 1000),
    );

  // Pattern selection
  const patternNames = PRESET_PATTERNS.reduce(
    (acc, pattern) => {
      acc[pattern.name] = pattern.url;
      return acc;
    },
    {} as Record<string, string>,
  );

  gui
    .add(appState, 'selectedPattern', patternNames)
    .name('Pattern')
    .onChange(() => {
      appState.customPatternFile = null;
      generateAutostereogram();
    });

  // Custom pattern file input
  const customPatternInput = document.createElement('input');
  customPatternInput.type = 'file';
  customPatternInput.accept = 'image/*';
  customPatternInput.style.display = 'none';
  document.body.appendChild(customPatternInput);

  customPatternInput.addEventListener('change', (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (file) {
      appState.customPatternFile = file;
      generateAutostereogram();
    }
  });

  gui
    .add(
      {uploadCustomPattern: () => customPatternInput.click()},
      'uploadCustomPattern',
    )
    .name('Upload Custom Pattern');

  // Re-render button
  gui.add({render: () => generateAutostereogram()}, 'render').name('Re-render');
}

async function main() {
  hide('preloader');

  const hasWebGPU = navigator.gpu !== undefined;

  const hasFp16 =
    hasWebGPU &&
    (async () => {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        return adapter!.features.has('shader-f16');
      } catch {
        return false;
      }
    })();

  show('loader');
  const depthEstimator = await pipeline(
    'depth-estimation',
    'onnx-community/depth-anything-v2-small',
    {
      dtype: hasFp16 ? 'fp16' : undefined,
      device: hasWebGPU ? 'webgpu' : undefined,
    },
  );
  hide('loader');

  // Setup GUI controls (initially hidden)
  setupGUI();

  show('image-chooser');

  // Wait for the user to choose a photo...
  const imageChooser = document.getElementById(
    'choose-a-photo',
  ) as HTMLInputElement;

  // When the user chooses a photo, load it into the canvas
  imageChooser.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    // Hide GUI while processing new image
    appState.gui?.hide();

    hide('image-chooser');
    show('loading-depth-estimation');

    try {
      const image = await RawImage.fromBlob(file);
      appState.currentImage = image;

      const {depth} = (await depthEstimator(
        image,
      )) as DepthEstimationPipelineOutput;

      const canvasElement = document.getElementById(
        'canvas',
      ) as HTMLCanvasElement;
      const hiddenImageCanvas = new OffscreenCanvas(
        canvasElement.width,
        canvasElement.height,
      );
      const hiddenImageCtx = hiddenImageCanvas.getContext('2d')!;
      {
        hiddenImageCtx.fillStyle = 'black';
        hiddenImageCtx.fillRect(
          0,
          0,
          hiddenImageCanvas.width,
          hiddenImageCanvas.height,
        );
        // Draw the image centered on the canvas
        drawImageCentered(depth.toCanvas(), hiddenImageCanvas);
      }

      // Store the depth data in app state
      appState.currentDepth = new PixelGrid(
        hiddenImageCtx.getImageData(
          0,
          0,
          hiddenImageCanvas.width,
          hiddenImageCanvas.height,
        ),
      );

      // Generate the initial autostereogram
      await generateAutostereogram();

      // Show the GUI controls now that we have an image
      appState.gui?.show();

      hide('messages');

      // Add save image functionality (only add once)
      const saveButton = document.getElementById(
        'save-image',
      ) as HTMLButtonElement;

      // Remove existing event listeners to prevent duplicates
      const newSaveButton = saveButton.cloneNode(true) as HTMLButtonElement;
      saveButton.parentNode?.replaceChild(newSaveButton, saveButton);

      newSaveButton.addEventListener('click', () => {
        const canvas = document.getElementById('canvas') as HTMLCanvasElement;
        const link = document.createElement('a');
        link.download = 'autostereogram.png';
        link.href = canvas.toDataURL();
        link.click();
      });
    } catch (error) {
      console.error('Error processing image:', error);
      appState.gui?.hide();
      hide('loading-depth-estimation');
      show('image-chooser');
    }
  });
}

main();

/**
 * Draws an image centered on a canvas with the correct aspect ratio, fitting
 * the entire picture.
 */
function drawImageCentered(
  /** The RawImage to draw */
  image: HTMLCanvasElement | OffscreenCanvas,
  /** The canvas to draw on */
  canvas: HTMLCanvasElement | OffscreenCanvas,
) {
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

  // Get dimensions
  const imageWidth = image.width;
  const imageHeight = image.height;
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;

  // Calculate scale to fit the entire image
  const scaleX = canvasWidth / imageWidth;
  const scaleY = canvasHeight / imageHeight;
  const scale = Math.min(scaleX, scaleY);

  // Calculate centered position
  const scaledWidth = imageWidth * scale;
  const scaledHeight = imageHeight * scale;
  const x = (canvasWidth - scaledWidth) / 2;
  const y = (canvasHeight - scaledHeight) / 2;

  ctx.drawImage(image, x, y, scaledWidth, scaledHeight);
}

/**
 * Fills the canvas with the image, as a repeating pattern
 */
function fillImage(
  image: HTMLCanvasElement | OffscreenCanvas,
  canvas: HTMLCanvasElement | OffscreenCanvas,
  /**
   * The width of the tiles to use for the pattern; the height will be
   * calculated to maintain the aspect ratio.
   */
  tileWidth: number,
) {
  const ctx = canvas.getContext('2d') as
    | CanvasRenderingContext2D
    | OffscreenCanvasRenderingContext2D;

  // Calculate tile height to maintain aspect ratio
  const tileHeight = (tileWidth * image.height) / image.width;

  // Calculate how many tiles we need to fill the canvas
  const tilesX = Math.ceil(canvas.width / tileWidth);
  const tilesY = Math.ceil(canvas.height / tileHeight);

  // Draw the image as a repeating pattern
  for (let y = 0; y < tilesY; y++) {
    for (let x = 0; x < tilesX; x++) {
      ctx.drawImage(
        image,
        x * tileWidth,
        y * tileHeight,
        tileWidth,
        tileHeight,
      );
    }
  }
}

type UiElementId =
  | 'messages'
  | 'title'
  | 'preloader'
  | 'loader'
  | 'warning'
  | 'run-demo'
  | 'exit-demo'
  | 'image-chooser'
  | 'loading-depth-estimation'
  | 'canvas'
  | 'save-image-container';

function hide(id: UiElementId) {
  document.getElementById(id)!.hidden = true;
}

function show(id: UiElementId) {
  document.getElementById(id)!.hidden = false;
}

function debounce(func: () => void, delay: number) {
  let timeout: NodeJS.Timeout;
  return () => {
    clearTimeout(timeout);
    timeout = setTimeout(func, delay);
  };
}
