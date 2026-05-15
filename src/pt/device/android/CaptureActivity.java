package com.pt.capture;

import android.app.Activity;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;
import android.view.SurfaceView;
import android.view.SurfaceHolder;
import android.view.ViewGroup;
import android.view.WindowManager;
import android.widget.FrameLayout;
import java.io.File;
import java.io.FileOutputStream;

public class CaptureActivity extends Activity implements SurfaceHolder.Callback {
    private static final String TAG = "PTCapture";
    private Camera mCamera;
    private SurfaceView mSurfaceView;
    private SurfaceHolder mHolder;
    private String mFileName;
    private boolean mCaptureStarted = false;
    private boolean mIsResumed = false;
    private boolean mSurfaceCreated = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        Log.i(TAG, "Lifecycle: onCreate");
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        mSurfaceView = new SurfaceView(this);
        mHolder = mSurfaceView.getHolder();
        mHolder.addCallback(this);

        FrameLayout layout = new FrameLayout(this);
        layout.addView(mSurfaceView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 
                ViewGroup.LayoutParams.MATCH_PARENT));
        setContentView(layout);

        mFileName = getIntent().getStringExtra("name");
        if (mFileName == null) mFileName = "manual_" + System.currentTimeMillis() + ".jpg";
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
            Log.i(TAG, "Conditions met, starting capture sequence...");
            runCapture();
        }
    }

    private void runCapture() {
        try {
            Log.i(TAG, "Opening camera...");
            mCamera = Camera.open(0);
            mCamera.setPreviewDisplay(mHolder);
            mCamera.startPreview();
            
            new Handler(Looper.getMainLooper()).postDelayed(new Runnable() {
                @Override
                public void run() {
                    takePicture();
                }
            }, 3000);

        } catch (Exception e) {
            Log.e(TAG, "Capture failed: " + e.getMessage());
            safeFinish();
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
