package com.pt.capture;

import android.app.Activity;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.content.Intent;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CaptureRequest;
import android.media.ImageReader;
import android.media.Image;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;

public class CaptureActivity extends Activity {
    private static final String TAG = "PTCapture";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "PT Capture activity started");
        new Thread(() -> runCapture()).start();
    }

    private void runCapture() {
        try {
            Intent intent = getIntent();
            String name = intent.getStringExtra("name");
            String mode = intent.getStringExtra("mode"); // "jpg" or "raw"
            int delay = intent.getIntExtra("delay", 5000); // milliseconds
            float exposure = intent.getFloatExtra("exposure", -1); // -1 = auto
            int iso = intent.getIntExtra("iso", -1); // -1 = auto

            if (name == null) {
                name = "pt_" + System.currentTimeMillis();
            }

            Log.i(TAG, "Starting capture: " + name + " mode=" + mode + " delay=" + delay + "ms");

            // Safety delay
            Thread.sleep(delay);

            if ("raw".equals(mode)) {
                captureRaw(name);
            } else {
                captureJpg(name, exposure, iso);
            }

        } catch (Exception e) {
            Log.e(TAG, "Capture failed: " + e.getMessage());
            finish();
        }
    }

    private void captureJpg(String name, float exposure, int iso) {
        try {
            Camera camera = Camera.open();
            Log.i(TAG, "Camera opened for JPG capture");

            Camera.Parameters params = camera.getParameters();

            // Configure exposure and ISO if specified
            if (exposure != -1) {
                params.setExposureCompensation((int)exposure);
            }
            if (iso != -1) {
                params.setIsoSpeed(iso);
            }

            camera.setParameters(params);
            camera.startPreview();
            Thread.sleep(300);

            camera.takePicture(null, null, (data, cam) -> {
                try {
                    File dir = new File(Environment.getExternalStorageDirectory(), "PTCaptures");
                    if (!dir.exists()) dir.mkdirs();

                    File file = new File(dir, name + ".jpg");
                    FileOutputStream fos = new FileOutputStream(file);
                    fos.write(data);
                    fos.close();

                    Log.i(TAG, "JPG saved: " + file.getAbsolutePath());
                    Log.i(TAG, "CAPTURE_COMPLETE:" + name + ".jpg");

                    cam.release();
                    finish();

                } catch (Exception e) {
                    Log.e(TAG, "Error saving JPG: " + e.getMessage());
                    cam.release();
                    finish();
                }
            });

        } catch (Exception e) {
            Log.e(TAG, "JPG capture failed: " + e.getMessage());
            finish();
        }
    }

    private void captureRaw(String name) {
        // RAW capture using Camera2 API (for devices that support it)
        try {
            Log.i(TAG, "RAW capture not implemented yet - falling back to JPG");
            captureJpg(name, -1, -1);
        } catch (Exception e) {
            Log.e(TAG, "RAW capture failed: " + e.getMessage());
            finish();
        }
    }
}