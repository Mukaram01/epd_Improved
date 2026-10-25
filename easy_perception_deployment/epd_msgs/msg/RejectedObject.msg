# A detection EPD did not localize: its 3D geometry failed EPD's geometry validation.
string name  # detected class
float64 confidence  # detector score (0..1)
sensor_msgs/RegionOfInterest roi  # 2D bounding box of the detection
uint8 DEGRADED=1
uint8 INVALID=2
uint8 quality  # EPD GeometryQuality of the rejected detection
# Bit set of EPD GeometryFailure: 1 intrinsics, 2 ROI, 4 mask, 8 insufficient depth, 16 empty cloud,
# 32 non-finite geometry, 64 dimensions, 128 orientation, 256 frame mismatch, 512 geometry exception.
uint32 failure_reasons
uint32 mask_pixel_count
uint32 valid_depth_pixel_count
float64 valid_depth_ratio
