package com.pt.capture;

import android.app.Activity;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.util.Range;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.os.StatFs;
import android.os.Build;
import android.content.Context;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.params.StreamConfigurationMap;
import java.io.File;
import java.io.FileOutputStream;
import java.util.List;

public class CaptureActivity extends Activity implements SurfaceHolder.Callback {
    private static final String TAG = "PTCapture";
    private Camera mCamera;
    private SurfaceView mSurfaceView;
    private SurfaceHolder mHolder;
    private String mFileName;
    private int mZoomPercent = 0;
    private int mDelayMs = 5000;
    private int mExposureCompensation = 0;
    private String mIso = "auto";
    private String mFocusMode = "continuous-picture";
    private String mAntibanding = "60hz";
    private String mWhiteBalance = "daylight";
    private String mMode = "jpg";
    private boolean mCaptureStarted = false;
    private boolean mIsResumed = false;
    private boolean mSurfaceCreated = false;

    private long getAvailableStorage() {
        try {
            File path = Environment.getExternalStorageDirectory();
            StatFs stat = new StatFs(path.getPath());
            long blockSize, availableBlocks;
            
            if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.JELLY_BEAN_MR2) {
                blockSize = stat.getBlockSizeLong();
                availableBlocks = stat.getAvailableBlocksLong();
            } else {
                blockSize = (long) stat.getBlockSize();
                availableBlocks = (long) stat.getAvailableBlocks();
            }
            return availableBlocks * blockSize;
        } catch (Exception e) {
            Log.e(TAG, "Failed to check storage: " + e.getMessage());
            return -1;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "Lifecycle: onCreate");
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON |
                            WindowManager.LayoutParams.FLAG_DISMISS_KEYGUARD |
                            WindowManager.LayoutParams.FLAG_SHOW_WHEN_LOCKED |
                            WindowManager.LayoutParams.FLAG_TURN_SCREEN_ON);

        readIntentSettings();
        if ("probe".equals(mMode)) {
            writeCapabilityProbe();
            safeFinish();
            return;
        }

        mSurfaceView = new SurfaceView(this);
        mHolder = mSurfaceView.getHolder();
        mHolder.addCallback(this);

        FrameLayout layout = new FrameLayout(this);
        layout.addView(mSurfaceView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 
                ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(layout);

    }

    private void readIntentSettings() {
        mFileName = getIntent().getStringExtra("name");
        if (mFileName == null) mFileName = "manual_" + System.currentTimeMillis() + ".jpg";
        String mode = getIntent().getStringExtra("mode");
        if (mode != null) mMode = mode;
        mZoomPercent = getIntent().getIntExtra("zoomPercent", 0);
        mDelayMs = getIntent().getIntExtra("delay", 5000);
        mExposureCompensation = getIntent().getIntExtra("exposureCompensation", 0);
        String iso = getIntent().getStringExtra("iso");
        if (iso != null) mIso = iso;
        String focusMode = getIntent().getStringExtra("focusMode");
        if (focusMode != null) mFocusMode = focusMode;
        String antibanding = getIntent().getStringExtra("antibanding");
        if (antibanding != null) mAntibanding = antibanding;
        String whiteBalance = getIntent().getStringExtra("whiteBalance");
        if (whiteBalance != null) mWhiteBalance = whiteBalance;
    }

    @Override
    protected void onResume() {
        super.onResume();
        Log.i(TAG, "Lifecycle: onResume");
        mIsResumed = true;
        
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                tryToStartCapture();
            }
        });
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.i(TAG, "Lifecycle: onPause");
        mIsResumed = false;
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        Log.i(TAG, "Surface created");
        mSurfaceCreated = true;
        tryToStartCapture();
    }

    @Override public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {}
    @Override public void surfaceDestroyed(SurfaceHolder holder) {
        mSurfaceCreated = false;
    }

    private void tryToStartCapture() {
        if (mIsResumed && mSurfaceCreated && !mCaptureStarted) {
            mCaptureStarted = true;
            if ("probe".equals(mMode)) {
                writeCapabilityProbe();
                safeFinish();
                return;
            }
            Log.i(TAG, "Conditions met, starting capture sequence...");
            runCapture();
        }
    }

    private String jsonQuote(String value) {
        if (value == null) return "null";
        return "\"" + value.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private String listToJson(List<String> values) {
        if (values == null) return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) {
            if (i > 0) sb.append(",");
            sb.append(jsonQuote(values.get(i)));
        }
        sb.append("]");
        return sb.toString();
    }

    private void writeCapabilityProbe() {
        StringBuilder json = new StringBuilder();
        json.append("{");
        json.append("\"device_time_ms\":").append(System.currentTimeMillis()).append(",");
        json.append("\"sdk_int\":").append(Build.VERSION.SDK_INT).append(",");

        Camera camera = null;
        try {
            camera = Camera.open(0);
            Camera.Parameters params = camera.getParameters();
            json.append("\"camera1\":{");
            json.append("\"available\":true,");
            json.append("\"zoom_supported\":").append(params.isZoomSupported()).append(",");
            json.append("\"max_zoom\":").append(params.isZoomSupported() ? params.getMaxZoom() : 0).append(",");
            json.append("\"min_exposure_compensation\":").append(params.getMinExposureCompensation()).append(",");
            json.append("\"max_exposure_compensation\":").append(params.getMaxExposureCompensation()).append(",");
            json.append("\"exposure_compensation_step\":").append(params.getExposureCompensationStep()).append(",");
            json.append("\"focus_modes\":").append(listToJson(params.getSupportedFocusModes())).append(",");
            json.append("\"white_balance_modes\":").append(listToJson(params.getSupportedWhiteBalance())).append(",");
            json.append("\"antibanding_modes\":").append(listToJson(params.getSupportedAntibanding())).append(",");
            json.append("\"scene_modes\":").append(listToJson(params.getSupportedSceneModes())).append(",");
            json.append("\"color_effects\":").append(listToJson(params.getSupportedColorEffects()));
            json.append("},");
        } catch (Exception e) {
            json.append("\"camera1\":{\"available\":false,\"error\":").append(jsonQuote(e.getMessage())).append("},");
        } finally {
            if (camera != null) {
                try { camera.release(); } catch (Exception e) {}
            }
        }

        json.append("\"camera2\":");
        if (Build.VERSION.SDK_INT >= 21) {
            try {
                CameraManager manager = (CameraManager)getSystemService(Context.CAMERA_SERVICE);
                String[] ids = manager.getCameraIdList();
                json.append("{\"available\":true,\"cameras\":[");
                for (int i = 0; i < ids.length; i++) {
                    if (i > 0) json.append(",");
                    CameraCharacteristics c = manager.getCameraCharacteristics(ids[i]);
                    Integer level = c.get(CameraCharacteristics.INFO_SUPPORTED_HARDWARE_LEVEL);
                    int[] capabilities = c.get(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES);
                    Float minFocus = c.get(CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE);
                    Range<Integer> isoRange = c.get(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE);
                    Range<Long> exposureRange = c.get(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE);
                    StreamConfigurationMap map = c.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                    json.append("{");
                    json.append("\"id\":").append(jsonQuote(ids[i])).append(",");
                    json.append("\"hardware_level\":").append(level == null ? -1 : level).append(",");
                    json.append("\"minimum_focus_distance\":").append(minFocus == null ? 0 : minFocus).append(",");
                    json.append("\"iso_min\":").append(isoRange == null ? 0 : isoRange.getLower()).append(",");
                    json.append("\"iso_max\":").append(isoRange == null ? 0 : isoRange.getUpper()).append(",");
                    json.append("\"exposure_time_min_ns\":").append(exposureRange == null ? 0 : exposureRange.getLower()).append(",");
                    json.append("\"exposure_time_max_ns\":").append(exposureRange == null ? 0 : exposureRange.getUpper()).append(",");
                    json.append("\"manual_sensor\":").append(hasCapability(capabilities, CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR)).append(",");
                    json.append("\"manual_post_processing\":").append(hasCapability(capabilities, CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING)).append(",");
                    json.append("\"raw\":").append(hasCapability(capabilities, CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES_RAW)).append(",");
                    json.append("\"jpeg_output\":").append(map != null && map.getOutputSizes(android.graphics.ImageFormat.JPEG) != null);
                    json.append("}");
                }
                json.append("]}");
            } catch (Exception e) {
                json.append("{\"available\":false,\"error\":").append(jsonQuote(e.getMessage())).append("}");
            }
        } else {
            json.append("{\"available\":false,\"reason\":\"sdk_below_21\"}");
        }
        json.append("}");

        try {
            File dir = new File(Environment.getExternalStorageDirectory(), "PTCaptures");
            if (!dir.exists()) dir.mkdirs();
            File file = new File(dir, "capabilities.json");
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(json.toString().getBytes("UTF-8"));
            fos.flush();
            fos.close();
            Log.i(TAG, "CAPABILITY_PROBE:" + json.toString());
            Log.i(TAG, "CAPABILITY_COMPLETE:capabilities.json");
        } catch (Exception e) {
            Log.e(TAG, "Capability probe failed: " + e.getMessage());
        }
    }

    private boolean hasCapability(int[] capabilities, int target) {
        if (capabilities == null) return false;
        for (int value : capabilities) {
            if (value == target) return true;
        }
        return false;
    }

    private void runCapture() {
        try {
            long available = getAvailableStorage();
            if (available != -1 && available < 10 * 1024 * 1024) { // 10MB threshold
                Log.e(TAG, "CRITICAL: Storage Low (" + (available / (1024 * 1024)) + " MB)");
                safeFinish();
                return;
            }

            Log.i(TAG, "Opening camera...");
            mCamera = Camera.open(0);
            applyCameraSettings(mCamera);
            mCamera.setPreviewDisplay(mHolder);
            mCamera.startPreview();
            
            new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
                @Override
                public void run() {
                    takePicture();
                }
            }, Math.max(500, mDelayMs));

        } catch (Exception e) {
            Log.e(TAG, "Capture failed: " + e.getMessage());
            safeFinish();
        }
    }

    private void applyCameraSettings(Camera camera) {
        try {
            Camera.Parameters params = camera.getParameters();

            if (params.isZoomSupported()) {
                int maxZoom = params.getMaxZoom();
                int clampedPercent = Math.max(0, Math.min(100, mZoomPercent));
                int zoomValue = Math.round((maxZoom * clampedPercent) / 100.0f);
                params.setZoom(zoomValue);
                Log.i(TAG, "Zoom set: " + clampedPercent + "% (" + zoomValue + "/" + maxZoom + ")");
            } else {
                Log.i(TAG, "Zoom not supported on this device");
            }

            if (params.getSupportedFocusModes() != null && params.getSupportedFocusModes().contains(mFocusMode)) {
                params.setFocusMode(mFocusMode);
                Log.i(TAG, "Focus mode set: " + mFocusMode);
            } else if (params.getSupportedFocusModes() != null && params.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
                params.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
                Log.i(TAG, "Focus mode fallback: auto");
            }

            if (params.getSupportedAntibanding() != null && params.getSupportedAntibanding().contains(mAntibanding)) {
                params.setAntibanding(mAntibanding);
                Log.i(TAG, "Antibanding set: " + mAntibanding);
            }

            if (params.getSupportedWhiteBalance() != null && params.getSupportedWhiteBalance().contains(mWhiteBalance)) {
                params.setWhiteBalance(mWhiteBalance);
                Log.i(TAG, "White balance set: " + mWhiteBalance);
            } else if (params.getSupportedWhiteBalance() != null && params.getSupportedWhiteBalance().contains(Camera.Parameters.WHITE_BALANCE_DAYLIGHT)) {
                params.setWhiteBalance(Camera.Parameters.WHITE_BALANCE_DAYLIGHT);
                Log.i(TAG, "White balance fallback: daylight");
            } else {
                Log.i(TAG, "White balance not supported or unavailable: " + mWhiteBalance);
            }

            int minExposure = params.getMinExposureCompensation();
            int maxExposure = params.getMaxExposureCompensation();
            int exposure = Math.max(minExposure, Math.min(maxExposure, mExposureCompensation));
            params.setExposureCompensation(exposure);
            Log.i(TAG, "Exposure compensation set: " + exposure + " (" + minExposure + " to " + maxExposure + ")");

            if (!"auto".equals(mIso)) {
                params.set("iso", mIso);
                params.set("iso-speed", mIso);
                params.set("nv-picture-iso", mIso);
                Log.i(TAG, "ISO hint set: " + mIso);
            }

            params.setJpegQuality(95);
            camera.setParameters(params);
        } catch (Exception e) {
            Log.e(TAG, "Failed to apply camera settings: " + e.getMessage());
        }
    }

    private void takePicture() {
        try {
            if (mCamera == null) {
                safeFinish();
                return;
            }
            mCamera.takePicture(null, null, new Camera.PictureCallback() {
                @Override
                public void onPictureTaken(byte[] data, Camera camera) {
                    if (data != null) {
                        saveToFile(data);
                    }
                    safeFinish();
                }
            });
        } catch (Exception e) {
            Log.e(TAG, "Take picture failed: " + e.getMessage());
            safeFinish();
        }
    }

    private void saveToFile(byte[] data) {
        try {
            File dir = new File(Environment.getExternalStorageDirectory(), "PTCaptures");
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    Log.e(TAG, "Failed to create directory: " + dir.getAbsolutePath());
                }
            }

            String fullFileName = mFileName;
            if (!fullFileName.toLowerCase().endsWith(".jpg")) {
                fullFileName += ".jpg";
            }

            File file = new File(dir, fullFileName);
            FileOutputStream fos = new FileOutputStream(file);
            fos.write(data);
            fos.flush();
            fos.close();

            Log.i(TAG, "Photo saved: " + file.getAbsolutePath());
            Log.i(TAG, "CAPTURE_COMPLETE:" + fullFileName);
        } catch (Exception e) {
            Log.e(TAG, "Error saving photo: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private void safeFinish() {
        if (mCamera != null) {
            try {
                mCamera.stopPreview();
                mCamera.release();
            } catch (Exception e) {}
            mCamera = null;
        }
        
        Log.i(TAG, "Post-delayed finish call...");
        new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
            @Override
            public void run() {
                Log.i(TAG, "Executing finish()");
                finish();
            }
        }, 2000);
    }
}
