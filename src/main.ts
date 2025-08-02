import {
  DepthEstimationPipelineOutput,
  pipeline,
  RawImage,
} from '@huggingface/transformers';
import GUI from 'lil-gui';

import {
  generateConfettiPattern,
  generateNoisePattern,
  generateSprinklesPattern,
} from './PatternGenerator';
import {PixelGrid} from './PixelGrid';

// Available patterns in the public folder
const PRESET_PATTERNS = [
  {name: 'Flowers', url: 'vintage-flowers-sm.jpg'},
] as const;

type AppState = {
  disparityScale: number;
  selectedPattern:
    | (typeof PRESET_PATTERNS)[number]['url']
    | 'custom'
    | 'confetti'
    | 'sprinkles'
    | 'noise';
  customPatternFile: File | null;
  currentImage: RawImage | null;
  currentDepth: PixelGrid | null;
  originalDepthEstimation: RawImage | null;
  autostereogramImageData: ImageData | null;
  isProcessing: boolean;
  gui: GUI | null;
  fadeTimeout: NodeJS.Timeout | null;
  fadeAnimations: Animation[] | null;
  displayMode: 'autostereogram' | 'depth-map' | 'source-image';
  depthDisplayMode: DepthDisplayMode;
  watermark: string;
  gradientColor1: string;
  gradientColor2: string;
  gradientColor3: string;
  updateGradientControls?: () => void;
};

// Convert HSL to hex color
function hslToHex(h: number, s: number, l: number): string {
  s /= 100;
  l /= 100;

  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  let r = 0,
    g = 0,
    b = 0;

  if (0 <= h && h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (60 <= h && h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (120 <= h && h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (180 <= h && h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (240 <= h && h < 300) {
    r = x;
    g = 0;
    b = c;
  } else if (300 <= h && h < 360) {
    r = c;
    g = 0;
    b = x;
  }

  const rHex = Math.round((r + m) * 255)
    .toString(16)
    .padStart(2, '0');
  const gHex = Math.round((g + m) * 255)
    .toString(16)
    .padStart(2, '0');
  const bHex = Math.round((b + m) * 255)
    .toString(16)
    .padStart(2, '0');

  return `#${rHex}${gHex}${bHex}`;
}

// Generate random HSL colors with pleasingly spaced hues
function generateRandomGradientColors(): {
  color1: string;
  color2: string;
  color3: string;
} {
  const baseHue = Math.random() * 360;
  const color1 = hslToHex(baseHue, 70, 60);
  const color2 = hslToHex((baseHue + 120) % 360, 80, 80);
  const color3 = hslToHex((baseHue + 240) % 360, 70, 60);
  return {color1, color2, color3};
}

// Global state for autostereogram generation
const initialGradientColors = generateRandomGradientColors();
const appState: AppState = {
  disparityScale: 1,
  selectedPattern: 'noise',
  customPatternFile: null,
  currentImage: null,
  currentDepth: null,
  originalDepthEstimation: null,
  autostereogramImageData: null,
  isProcessing: false,
  gui: null,
  fadeTimeout: null,
  fadeAnimations: null,
  displayMode: 'autostereogram',
  depthDisplayMode: 'clamp',
  watermark: 'LOU.WTF',
  gradientColor1: initialGradientColors.color1,
  gradientColor2: initialGradientColors.color2,
  gradientColor3: initialGradientColors.color3,
};

/**
 * Generates an autostereogram from the current depth image and pattern
 */
async function generateAutostereogram(): Promise<void> {
  if (!appState.currentDepth || appState.isProcessing) {
    return;
  }

  appState.isProcessing = true;
  show('generating-autostereogram');
  show('messages');
  hide('canvas');
  hide('depth-canvas');

  const canvasElement = document.getElementById('canvas') as HTMLCanvasElement;
  const hiddenImageCanvas = new OffscreenCanvas(
    canvasElement.width,
    canvasElement.height,
  );

  const minDisparity = Math.floor(hiddenImageCanvas.width * 0.15);
  const maxDisparity = Math.floor(hiddenImageCanvas.width * 0.2);
  const tileWidth = minDisparity;

  // Load the pattern image
  let patternImage: HTMLCanvasElement;
  if (appState.customPatternFile) {
    const patternImageUrl = URL.createObjectURL(appState.customPatternFile);
    patternImage = (await RawImage.fromURL(patternImageUrl)).toCanvas();
    URL.revokeObjectURL(patternImageUrl);
  } else if (appState.selectedPattern === 'confetti') {
    patternImage = generateConfettiPattern(
      appState.gradientColor1,
      appState.gradientColor2,
      appState.gradientColor3,
    );
  } else if (appState.selectedPattern === 'sprinkles') {
    patternImage = generateSprinklesPattern(
      appState.gradientColor1,
      appState.gradientColor2,
      appState.gradientColor3,
    );
  } else if (appState.selectedPattern === 'noise') {
    patternImage = generateNoisePattern(
      appState.gradientColor1,
      appState.gradientColor2,
      appState.gradientColor3,
    );
  } else {
    patternImage = (
      await RawImage.fromURL(appState.selectedPattern)
    ).toCanvas();
  }
  const patternCanvas = new OffscreenCanvas(tileWidth, canvasElement.height);
  fillImage(patternImage, patternCanvas, tileWidth);
  {
    const ctx = patternCanvas.getContext('2d')!;
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.75)';
    ctx.lineWidth = 1;
    {
      ctx.font =
        '800 18px "Arial Black", "Helvetica Black", "Helvetica Neue", Helvetica, Arial, sans-serif';

      const stringWidth = ctx.measureText(appState.watermark).width;
      ctx.fillText(appState.watermark, 0, 40);
      ctx.strokeText(appState.watermark, 0, 40);

      ctx.fillText(
        appState.watermark,
        tileWidth - stringWidth,
        canvasElement.height - 40,
      );
      ctx.strokeText(
        appState.watermark,
        tileWidth - stringWidth,
        canvasElement.height - 40,
      );
    }
  }

  const pattern = new PixelGrid(
    patternCanvas
      .getContext('2d')!
      .getImageData(0, 0, patternCanvas.width, patternCanvas.height),
  );

  const outputCanvas = new OffscreenCanvas(
    hiddenImageCanvas.width,
    hiddenImageCanvas.height,
  );

  fillImage(patternCanvas, outputCanvas, tileWidth);

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

  // Store the autostereogram image data in app state
  appState.autostereogramImageData = output.imageData;

  // Display the appropriate canvas based on current setting
  updateCanvasDisplay();

  appState.isProcessing = false;
  hide('messages');
  hide('generating-autostereogram');
}

/**
 * Updates which canvas is displayed based on the displayMode setting
 */
function updateCanvasDisplay(): void {
  if (!appState.currentDepth) {
    return; // No depth data available yet
  }

  switch (appState.displayMode) {
    case 'depth-map': {
      hide('canvas');
      show('depth-canvas');

      // Regenerate depth canvas with current display mode and copy to visible
      // canvas
      regenerateDepthCanvasInternal();

      // Copy depth map to the visible canvas
      const depthCanvasElement = document.getElementById(
        'depth-canvas',
      ) as HTMLCanvasElement;
      const ctx = depthCanvasElement.getContext('2d')!;
      ctx.putImageData(appState.currentDepth.imageData, 0, 0);
      break;
    }

    case 'source-image': {
      hide('depth-canvas');
      show('canvas');

      // Draw the original source image
      if (appState.currentImage) {
        const canvasElement = document.getElementById(
          'canvas',
        ) as HTMLCanvasElement;
        const ctx = canvasElement.getContext('2d')!;

        // Clear canvas
        ctx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        // Draw the source image centered
        drawImageCentered(appState.currentImage.toCanvas(), canvasElement);
      }
      break;
    }

    case 'autostereogram':
    default: {
      hide('depth-canvas');
      show('canvas');

      // Copy autostereogram to the visible canvas
      if (appState.autostereogramImageData) {
        const canvasElement = document.getElementById(
          'canvas',
        ) as HTMLCanvasElement;
        const ctx = canvasElement.getContext('2d')!;
        ctx.putImageData(appState.autostereogramImageData, 0, 0);
      }
      break;
    }
  }
}

/**
 * Regenerates the depth canvas with the current depth display mode
 */
function regenerateDepthCanvas(): void {
  if (!appState.originalDepthEstimation || !appState.currentDepth) {
    return; // No depth estimation data available
  }

  regenerateDepthCanvasInternal();

  // Regenerate the autostereogram with the new depth data
  generateAutostereogram();
}

/**
 * Internal function to regenerate the depth canvas without triggering
 * autostereogram generation
 */
function regenerateDepthCanvasInternal(): void {
  if (!appState.originalDepthEstimation) {
    return; // No depth estimation data available
  }

  const canvasElement = document.getElementById('canvas') as HTMLCanvasElement;
  const hiddenImageCanvas = new OffscreenCanvas(
    canvasElement.width,
    canvasElement.height,
  );
  const hiddenImageCtx = hiddenImageCanvas.getContext('2d')!;

  // Clear the canvas with appropriate background based on depth display mode
  if (appState.depthDisplayMode === 'popout') {
    hiddenImageCtx.fillStyle = 'black';
  } else if (appState.depthDisplayMode === 'cutout') {
    hiddenImageCtx.fillStyle = 'white';
  } else {
    // clamp mode
    hiddenImageCtx.fillStyle = 'black';
  }

  hiddenImageCtx.fillRect(
    0,
    0,
    hiddenImageCanvas.width,
    hiddenImageCanvas.height,
  );

  // Get the original depth estimation result
  const depthCanvas =
    appState.originalDepthEstimation.toCanvas() as OffscreenCanvas;

  // Draw the depth image centered on the canvas with current depth display mode
  drawImageCentered(depthCanvas, hiddenImageCanvas, appState.depthDisplayMode);

  // Update the depth data in app state
  appState.currentDepth = new PixelGrid(
    hiddenImageCtx.getImageData(
      0,
      0,
      hiddenImageCanvas.width,
      hiddenImageCanvas.height,
    ),
  );
}

/**
 * Sets up the Advanced GUI controls
 */
function setupGUI(): void {
  const gui = new GUI();
  gui.close();
  gui.title('Advanced controls');

  // Hide the GUI initially
  gui.hide();

  // Store GUI reference in app state
  appState.gui = gui;

  // Disparity scale slider
  gui
    .add(appState, 'disparityScale', 0.1, 1.75, 0.01)
    .name('Depth')
    .onChange(
      debounce(() => {
        generateAutostereogram();
      }, 1000),
    );

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

  // Pattern selection with custom option
  const patternNames = PRESET_PATTERNS.reduce(
    (acc, pattern) => {
      acc[pattern.name] = pattern.url;
      return acc;
    },
    {} as Record<string, string>,
  );
  patternNames['Noise'] = 'noise';
  patternNames['Upload...'] = 'custom';

  gui
    .add(appState, 'selectedPattern', patternNames)
    .name('Pattern')
    .onChange(() => {
      if (appState.selectedPattern === 'custom') {
        customPatternInput.click();
      } else {
        appState.customPatternFile = null;
        generateAutostereogram();
      }
      updateGradientControls();
    });

  // Gradient color controls (initially hidden)
  const gradientFolder = gui.addFolder('Gradient Colors');
  gradientFolder.hide();

  gradientFolder
    .addColor(appState, 'gradientColor1')
    .name('Color 1')
    .onChange(
      debounce(() => {
        if (isGeneratedPattern(appState.selectedPattern)) {
          generateAutostereogram();
        }
      }, 500),
    );

  gradientFolder
    .addColor(appState, 'gradientColor2')
    .name('Color 2')
    .onChange(
      debounce(() => {
        if (isGeneratedPattern(appState.selectedPattern)) {
          generateAutostereogram();
        }
      }, 500),
    );

  gradientFolder
    .addColor(appState, 'gradientColor3')
    .name('Color 3')
    .onChange(
      debounce(() => {
        if (isGeneratedPattern(appState.selectedPattern)) {
          generateAutostereogram();
        }
      }, 500),
    );

  // Function to check if the selected pattern is a generated pattern
  function isGeneratedPattern(pattern: string): boolean {
    return (
      pattern === 'confetti' || pattern === 'sprinkles' || pattern === 'noise'
    );
  }

  // Function to update gradient controls visibility
  function updateGradientControls(): void {
    if (isGeneratedPattern(appState.selectedPattern)) {
      gradientFolder.show();
    } else {
      gradientFolder.hide();
    }
  }

  // Store the updateGradientControls function in app state for external access
  appState.updateGradientControls = updateGradientControls;

  // Display mode dropdown
  const displayModeOptions = {
    Autostereogram: 'autostereogram',
    'Depth map': 'depth-map',
    'Source image': 'source-image',
  };

  gui
    .add(appState, 'watermark')
    .name('Watermark')
    .onChange(
      debounce(() => {
        generateAutostereogram();
      }, 1000),
    );

  gui
    .add(appState, 'displayMode', displayModeOptions)
    .name('Display')
    .onChange(() => {
      updateCanvasDisplay();
    });

  // Depth display mode dropdown
  const depthDisplayModeOptions = {
    Clamp: 'clamp',
    Cutout: 'cutout',
    Popout: 'popout',
  };

  gui
    .add(appState, 'depthDisplayMode', depthDisplayModeOptions)
    .name('Depth Style')
    .onChange(() => {
      regenerateDepthCanvas();
    });
}

// Global variable for the image chooser input
let imageChooser: HTMLInputElement;

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

  // Setup GUI controls (initially hidden)
  setupGUI();

  // Setup standalone button event listeners
  const saveImageButton = document.getElementById(
    'save-image-button',
  ) as HTMLButtonElement;
  const chooseAnotherPhotoButton = document.getElementById(
    'choose-another-photo-button',
  ) as HTMLButtonElement;

  saveImageButton.addEventListener('click', () => {
    const canvas =
      appState.displayMode === 'depth-map'
        ? (document.getElementById('depth-canvas') as HTMLCanvasElement)
        : (document.getElementById('canvas') as HTMLCanvasElement);
    const link = document.createElement('a');
    link.download = `${appState.displayMode}.png`;
    link.href = canvas.toDataURL();
    link.click();
  });

  chooseAnotherPhotoButton.addEventListener('click', () => {
    imageChooser.click();
  });

  show('image-chooser');

  // Wait for the user to choose a photo...
  imageChooser = document.getElementById('choose-a-photo') as HTMLInputElement;

  // When the user chooses a photo, load it into the canvas
  imageChooser.addEventListener('change', async (e) => {
    const file = (e.target as HTMLInputElement).files?.[0];
    if (!file) return;

    // Hide GUI while processing new image
    appState.gui?.hide();

    // Hide canvases and show loading messages
    hide('canvas');
    hide('depth-canvas');
    show('messages');

    // Only hide image-chooser if this is the first time loading an image
    if (!appState.currentImage) {
      hide('image-chooser');
    }

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

    show('loading-depth-estimation');

    const image = await RawImage.fromBlob(file);
    appState.currentImage = image;

    const {depth} = (await depthEstimator(
      image,
    )) as DepthEstimationPipelineOutput;
    hide('loading-depth-estimation');

    // Store the original depth estimation for later regeneration
    appState.originalDepthEstimation = depth;

    const canvasElement = document.getElementById(
      'canvas',
    ) as HTMLCanvasElement;
    const hiddenImageCanvas = new OffscreenCanvas(
      canvasElement.width,
      canvasElement.height,
    );
    const hiddenImageCtx = hiddenImageCanvas.getContext('2d')!;
    hiddenImageCtx.fillStyle = 'black';
    hiddenImageCtx.fillRect(
      0,
      0,
      hiddenImageCanvas.width,
      hiddenImageCanvas.height,
    );

    const depthCanvas = depth.toCanvas() as OffscreenCanvas;

    // Draw the image centered on the canvas with current depth display mode
    drawImageCentered(
      depthCanvas,
      hiddenImageCanvas,
      appState.depthDisplayMode,
    );

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

    // Show the GUI controls and buttons now that we have an image
    appState.gui?.show();
    appState.updateGradientControls?.();
    show('viewing-tips-link');
    show('save-image-button');
    show('choose-another-photo-button');
  });
}

main();

type DepthDisplayMode = 'clamp' | 'cutout' | 'popout';

/**
 * Draws an image centered on a canvas with the correct aspect ratio, fitting
 * the entire picture. The left and right edges of the image should fill the
 * rest of the canvas.
 */
function drawImageCentered(
  /** The RawImage to draw */
  image: HTMLCanvasElement | OffscreenCanvas,
  /** The canvas to draw on */
  canvas: HTMLCanvasElement | OffscreenCanvas,
  depthDisplayMode: null | DepthDisplayMode = null,
) {
  const ctx = canvas.getContext('2d') as CanvasRenderingContext2D;

  // Get dimensions
  const imageWidth = image.width;
  const imageHeight = image.height;
  const canvasWidth = canvas.width;
  const canvasHeight = canvas.height;
  const padding =
    depthDisplayMode === 'cutout' || depthDisplayMode === 'popout'
      ? canvasWidth * 0.1
      : 0;

  // Calculate scale to fit the entire image
  const scaleX = (canvasWidth - padding) / imageWidth;
  const scaleY = (canvasHeight - padding) / imageHeight;
  const scale = Math.min(scaleX, scaleY);

  // Calculate centered position
  const scaledWidth = imageWidth * scale;
  const scaledHeight = imageHeight * scale;
  const x = (canvasWidth - scaledWidth) / 2;
  const y = (canvasHeight - scaledHeight) / 2;

  if (depthDisplayMode === 'popout') {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
  } else if (depthDisplayMode === 'cutout') {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
  }

  const originalGlobalCompositeOperation = ctx.globalCompositeOperation;

  ctx.drawImage(image, x, y, scaledWidth, scaledHeight);

  if (depthDisplayMode === 'clamp') {
    // Extend the left edge of the image
    ctx.drawImage(
      image,
      // sx,
      0,
      // sy,
      y,
      // sWidth,
      1,
      // sHeight,
      imageHeight,
      // dx,
      0,
      // dy,
      0,
      // dWidth,
      x + 1,
      // dHeight
      canvasHeight,
    );

    // Extend the right edge of the image
    ctx.drawImage(
      image,
      // sx,
      imageWidth - 1,
      // sy,
      y,
      // sWidth,
      1,
      // sHeight,
      imageHeight,
      // dx,
      x + scaledWidth - 1,
      // dy,
      0,
      // dWidth,
      x + 1,
      // dHeight
      canvasHeight,
    );

    // Fade the left and right edges of the image with a horizontal gradient
    // with a left, center, and right stop:
    const gradient = ctx.createLinearGradient(0, 0, canvasWidth, 0);
    gradient.addColorStop(0, '#555');
    gradient.addColorStop(x / canvasWidth, 'white');
    gradient.addColorStop((canvasWidth - x) / canvasWidth, 'white');
    gradient.addColorStop(1, '#555');

    ctx.fillStyle = gradient;
    ctx.globalCompositeOperation = 'multiply';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);
  }

  // Reset blend mode to default
  ctx.globalCompositeOperation = originalGlobalCompositeOperation;
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
  | 'generating-autostereogram'
  | 'canvas'
  | 'depth-canvas'
  | 'viewing-tips-link'
  | 'save-image-button'
  | 'choose-another-photo-button';

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

function handleError(event: Event) {
  console.error(event);
  alert('An unexpected error occurred. Refresh the page to try again.');
}

window.addEventListener('error', handleError);
window.addEventListener('unhandledrejection', handleError);

let fadeTimeout: NodeJS.Timeout;
function cancelFade() {
  clearTimeout(fadeTimeout);
  document.body.classList.remove('gui-faded');
  fadeTimeout = setTimeout(fadeOutGUI, 2000);
}

function fadeOutGUI() {
  document.body.classList.add('gui-faded');
}
setTimeout(fadeOutGUI, 2000);

document.addEventListener('click', cancelFade);
document.addEventListener('mousemove', cancelFade);
document.addEventListener('keydown', cancelFade);
document.addEventListener('touchstart', cancelFade);
document.addEventListener('touchend', cancelFade);
document.addEventListener('touchmove', cancelFade);
document.addEventListener('touchcancel', cancelFade);
document.addEventListener('touchleave', cancelFade);
document.addEventListener('touchcancel', cancelFade);
