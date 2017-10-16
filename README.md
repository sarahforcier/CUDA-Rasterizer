CUDA Rasterizer
===============

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 4**

Sarah Forcier

Tested on GeForce GTX 1070

### Overview

![](img/rasterizer.mp4)

| Normal | Position | Depth | UV |
| ----------- | ----------- | ----------- | ----------- |
| ![](img/duck_normal.png) | ![](img/duck_position.png) | ![](img/duck_depth.png) | ![](img/duck_uv.png) |

### Features

#### Materials

##### Lambertian

| Cow | Duck | Truck |
| ----------- | ----------- | ----------- |
| ![](img/cow_lamb.png) | ![](img/duck_lamb.png) | ![](img/truck_lamb.png) |

##### Blinn-Phong

| Cow | Duck | Truck |
| ----------- | ----------- | ----------- |
| ![](img/cow_blinn.png) | ![](img/duck_blinn.png) | ![](img/truck_blinn.png) |

##### Textures

| Default UV mapping | with Perspective Correction | with Bilinear Filtering |
| ----------- | ----------- | ----------- |
| ![](img/texture_default.png) | ![](img/texture_corrected.png) | ![](img/texture_bilinear.png) |

#### Post Processing
##### Gaussian Blur
| Cow | Duck | Truck |
| ----------- | ----------- | ----------- |
| ![](img/cow_blur.png) | ![](img/duck_blur.png) | ![](img/truck_blur.png) |

##### Bloom
| Cow | Duck | Truck |
| ----------- | ----------- | ----------- |
| ![](img/cow_bloom.png) | ![](img/duck_bloom.png) | ![](img/truck_bloom.png) |

### Performance

![](img/performance.png)

### Credits

* [tinygltfloader](https://github.com/syoyo/tinygltfloader) by [@soyoyo](https://github.com/syoyo)
* [glTF Sample Models](https://github.com/KhronosGroup/glTF/blob/master/sampleModels/README.md)
