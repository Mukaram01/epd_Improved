@rem Environment hook: prepend the vendored ONNX Runtime binary directory to PATH.
call ament_prepend_unique_value PATH "%AMENT_CURRENT_PREFIX%\opt\epd_onnxruntime_vendor\bin"
