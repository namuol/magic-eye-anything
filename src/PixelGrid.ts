type Pixel = [number, number, number, number];

export class PixelGrid {
  imageData: ImageData;
  constructor(imageData: ImageData) {
    this.imageData = imageData;
  }
  get data() {
    return this.imageData.data;
  }
  get width() {
    return this.imageData.width;
  }
  get height() {
    return this.imageData.height;
  }

  /** Get pixel as [r, g, b, a]  */
  get(x: number, y: number): Pixel {
    const index = (y * this.width + x) * 4;
    return [
      this.data[index + 0] ?? 0x00,
      this.data[index + 1] ?? 0x00,
      this.data[index + 2] ?? 0x00,
      this.data[index + 3] ?? 0xff,
    ];
  }

  // Set pixel with r, g, b, a values
  set(x: number, y: number, pixel: Pixel) {
    const index = (y * this.width + x) * 4;
    this.data[index + 0] = pixel[0];
    this.data[index + 1] = pixel[1];
    this.data[index + 2] = pixel[2];
    this.data[index + 3] = pixel[3];
  }
}
