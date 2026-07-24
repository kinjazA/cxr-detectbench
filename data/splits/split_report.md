# Phase 3 Split Report

## Split Summary

| split | images | abnormal_images | normal_images | abnormal_rate |
|---|---:|---:|---:|---:|
| train | 10500 | 3076 | 7424 | 0.2930 |
| val | 2250 | 659 | 1591 | 0.2929 |
| test | 2250 | 659 | 1591 | 0.2929 |

## Class Image Distribution

| class_id | total | train | val | test |
|---|---:|---:|---:|---:|
| 0 | 3067 | 2147 | 460 | 460 |
| 1 | 186 | 130 | 28 | 28 |
| 2 | 452 | 316 | 68 | 68 |
| 3 | 2300 | 1610 | 345 | 345 |
| 4 | 353 | 247 | 53 | 53 |
| 5 | 386 | 270 | 58 | 58 |
| 6 | 613 | 429 | 92 | 92 |
| 7 | 1322 | 926 | 198 | 198 |
| 8 | 826 | 578 | 124 | 124 |
| 9 | 1134 | 794 | 170 | 170 |
| 10 | 1032 | 722 | 155 | 155 |
| 11 | 1981 | 1387 | 297 | 297 |
| 12 | 96 | 68 | 14 | 14 |
| 13 | 1617 | 1132 | 242 | 243 |

## Balance Diagnostics

- Max per-class image split-rate deviation: 0.0083 (class 12, train)
- Max per-class annotation split-rate deviation: 0.0385 (class 2, val)
- Image-level balance is the acceptance criterion for this split. Annotation-level balance is reported as a diagnostic because one image can contain multiple boxes.

## BBox Size Summary

Values are normalized by image width/height. Empty values mean image metadata was unavailable.

| class_id | boxes | w_p50 | h_p50 | area_p50 |
|---|---:|---:|---:|---:|
| 0 | 7162 | 0.1177 | 0.1162 | 0.0137 |
| 1 | 279 | 0.1996 | 0.1932 | 0.0374 |
| 2 | 960 | 0.0642 | 0.0652 | 0.0038 |
| 3 | 5427 | 0.4349 | 0.1348 | 0.0583 |
| 4 | 556 | 0.1819 | 0.1821 | 0.0323 |
| 5 | 1000 | 0.2193 | 0.3514 | 0.0735 |
| 6 | 1247 | 0.1810 | 0.1844 | 0.0337 |
| 7 | 2483 | 0.1550 | 0.1427 | 0.0211 |
| 8 | 2580 | 0.0399 | 0.0347 | 0.0014 |
| 9 | 2203 | 0.1156 | 0.1254 | 0.0168 |
| 10 | 2476 | 0.0874 | 0.1012 | 0.0088 |
| 11 | 4842 | 0.0929 | 0.0534 | 0.0044 |
| 12 | 226 | 0.2684 | 0.2813 | 0.0724 |
| 13 | 4655 | 0.1179 | 0.0951 | 0.0106 |

## Grouping

No patient/study group column was used. The current local metadata does not expose a patient_id/study_id key, so this split is image-level stratified.

Metadata columns seen: `PatientSex, PatientSize, PatientWeight, SamplesPerPixel, PhotometricInterpretation, Rows, Columns, PixelAspectRatio, BitsAllocated, BitsStored, HighBit, PixelRepresentation, WindowCenter, WindowWidth, RescaleIntercept, RescaleSlope, LossyImageCompression, fname, MultiPixelAspectRatio, PixelAspectRatio1, img_min, img_max, img_mean, img_std, img_pct_window, PatientAge, NumberOfFrames, PixelSpacing, LossyImageCompressionRatio, LossyImageCompressionMethod, MultiPixelSpacing, PixelSpacing1, SmallestImagePixelValue, LargestImagePixelValue`
