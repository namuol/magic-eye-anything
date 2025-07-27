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
  fadeTimeout: null as NodeJS.Timeout | null,
  fadeAnimation: null as Animation | null,
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
  gui.close();
  gui.title('Controls');

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
    });

  // Save image button
  gui
    .add(
      {
        saveImage: () => {
          const canvas = document.getElementById('canvas') as HTMLCanvasElement;
          const link = document.createElement('a');
          link.download = 'autostereogram.png';
          link.href = canvas.toDataURL();
          link.click();
        },
      },
      'saveImage',
    )
    .name('Save Image');

  // Add event listeners for fade functionality
  const guiElement = gui.domElement;

  // Reset fade timer on mouse enter
  guiElement.addEventListener('mouseenter', resetFadeTimer);

  // Reset fade timer on any interaction with GUI elements
  guiElement.addEventListener('click', resetFadeTimer);
  guiElement.addEventListener('input', resetFadeTimer);
  guiElement.addEventListener('change', resetFadeTimer);
  guiElement.addEventListener('touchstart', resetFadeTimer);
  guiElement.addEventListener('touchend', resetFadeTimer);

  // Start fade timer when GUI is shown
  const originalShow = gui.show.bind(gui);
  gui.show = (show?: boolean) => {
    const result = originalShow(show);
    if (show !== false) {
      startFadeTimer();
    }
    return result;
  };

  // Clear fade timer when GUI is hidden
  const originalHide = gui.hide.bind(gui);
  gui.hide = () => {
    const result = originalHide();
    if (appState.fadeTimeout) {
      clearTimeout(appState.fadeTimeout);
      appState.fadeTimeout = null;
    }
    if (appState.fadeAnimation) {
      appState.fadeAnimation.cancel();
      appState.fadeAnimation = null;
    }
    return result;
  };

  // Watch for expanded state changes
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (
        mutation.type === 'attributes' &&
        mutation.attributeName === 'class'
      ) {
        const target = mutation.target as Element;
        if (target.classList.contains('expanded')) {
          // GUI is expanded, cancel fade
          if (appState.fadeTimeout) {
            clearTimeout(appState.fadeTimeout);
            appState.fadeTimeout = null;
          }
          if (appState.fadeAnimation) {
            appState.fadeAnimation.cancel();
            appState.fadeAnimation = null;
          }
          guiElement.style.opacity = '1';
        } else {
          // GUI is collapsed, start fade timer
          startFadeTimer();
        }
      }
    });
  });

  observer.observe(guiElement, {attributes: true, attributeFilter: ['class']});

  // Add global mouse move listener to reset fade timer when mouse is near GUI
  document.addEventListener('mousemove', (e) => {
    if (!appState.gui) return;

    const rect = guiElement.getBoundingClientRect();
    const margin = 50; // 50px margin around GUI

    if (
      e.clientX >= rect.left - margin &&
      e.clientX <= rect.right + margin &&
      e.clientY >= rect.top - margin &&
      e.clientY <= rect.bottom + margin
    ) {
      resetFadeTimer();
    }
  });

  document.addEventListener('touchstart', resetFadeTimer);
  document.addEventListener('click', resetFadeTimer);
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
  | 'canvas';

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

/**
 * Starts the fade-out timer for the GUI controls
 */
function startFadeTimer(): void {
  if (!appState.gui) return;
  if (!appState.gui.domElement.classList.contains('closed')) return;

  // Clear any existing fade timeout
  if (appState.fadeTimeout) {
    clearTimeout(appState.fadeTimeout);
  }

  // Clear any existing fade animation
  if (appState.fadeAnimation) {
    appState.fadeAnimation.cancel();
  }

  // Check if GUI is expanded (has the 'expanded' class)
  const guiElement = appState.gui.domElement;
  if (guiElement.classList.contains('expanded')) {
    return; // Don't fade if expanded
  }

  // Set opacity to 1 (fully visible)
  guiElement.style.opacity = '1';

  // Start fade timer
  appState.fadeTimeout = setTimeout(() => {
    fadeOutGUI();
  }, 3000); // 3 second delay
}

/**
 * Fades out the GUI controls over 2 seconds
 */
function fadeOutGUI(): void {
  if (!appState.gui) return;

  const guiElement = appState.gui.domElement;

  // Don't fade if expanded
  if (guiElement.classList.contains('expanded')) {
    return;
  }

  // Create fade animation
  appState.fadeAnimation = guiElement.animate(
    [{opacity: '1'}, {opacity: '0.1'}],
    {
      duration: 2000, // 2 seconds
      easing: 'ease-out',
      fill: 'forwards',
    },
  );
}

/**
 * Resets the fade timer and makes GUI fully visible
 */
function resetFadeTimer(): void {
  if (!appState.gui) return;

  // Clear existing timeout and animation
  if (appState.fadeTimeout) {
    clearTimeout(appState.fadeTimeout);
    appState.fadeTimeout = null;
  }

  if (appState.fadeAnimation) {
    appState.fadeAnimation.cancel();
    appState.fadeAnimation = null;
  }

  // Make GUI fully visible
  appState.gui.domElement.style.opacity = '1';

  // Start new fade timer
  startFadeTimer();
}
