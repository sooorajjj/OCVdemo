package com.vst.demo.ocvdemo;

import android.Manifest;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

import static org.opencv.core.Core.NORM_MINMAX;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    private static final String TAG = MainActivity.class.getName();
    JavaCameraView javaCameraView;
    /**
     * the template image to use
     */
    private static final int TEMPLATE_IMAGE = R.drawable.template_switchboard2;
    /**
     * the result matrix
     */
    Mat result;
    /**
     * the camera image
     */
    Mat mCameraMat;
    /**
     * the template image used for template matching
     * or for copying into the camera view
     */
    Mat mTemplateMat;
//
//    /**
//     * the crop rectangle with the size of the template image
//     */
//    Rect rect;
//    /**
//     * selected area is the camera preview cut to the crop rectangle
//     */
//    Mat selectedArea;

    /**
     * selected area is the camera preview cut to the crop rectangle
     */
    private final double threshold = 1.0;
    /**
     * frame size width
     */
    private static final int FRAME_SIZE_WIDTH = 640;
    /**
     * frame size height
     */
    private static final int FRAME_SIZE_HEIGHT = 480;
    /**
     * whether or not to use a fixed frame size -> results usually in higher FPS
     * 640 x 480
     */
    private static final boolean FIXED_FRAME_SIZE = true;

    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case BaseLoaderCallback.SUCCESS:{

                    // load the specified image from file system in bgr color
                    Mat bgr = null;
                    try {
                        bgr = Utils.loadResource(getApplicationContext(), TEMPLATE_IMAGE, Imgcodecs.CV_LOAD_IMAGE_COLOR);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // convert the image to rgba
                    mTemplateMat = new Mat();
                    Imgproc.cvtColor(bgr, mTemplateMat, Imgproc.COLOR_BGR2RGBA);

                    result = new Mat();
                    javaCameraView.enableView();
                    break;
                }default:{
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    private boolean checkIfAlreadyhavePermission() {
        int result = ContextCompat.checkSelfPermission(this, Manifest.permission.GET_ACCOUNTS);
        if (result == PackageManager.PERMISSION_GRANTED) {
            return true;
        } else {
            return false;
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_PORTRAIT);

        //Requesting Permission
        int MyVersion = Build.VERSION.SDK_INT;
        if (MyVersion > Build.VERSION_CODES.LOLLIPOP_MR1) {
            if (!checkIfAlreadyhavePermission()) {
                requestForSpecificPermission();
            }
        }

        javaCameraView = (JavaCameraView) findViewById(R.id.java_camera_view);
        if (FIXED_FRAME_SIZE) {
            javaCameraView.setMaxFrameSize(FRAME_SIZE_WIDTH, FRAME_SIZE_HEIGHT);
        }
        javaCameraView.setVisibility(SurfaceView.VISIBLE);
        javaCameraView.setCvCameraViewListener(this);
    }

    private void requestForSpecificPermission() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 101);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case 101:
                if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    //granted

                } else {
                    //not granted
                    Toast.makeText(MainActivity.this, "Permission denied to access Camera", Toast.LENGTH_SHORT).show();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }
    @Override
    protected void onPause(){
        super.onPause();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(javaCameraView!=null){
            javaCameraView.disableView();
        }
    }

    @Override
    protected void onResume(){
        super.onResume();

        if(OpenCVLoader.initDebug()){
            Log.d(TAG, "OpenCV successfully loaded");

            mLoaderCallBack.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else
        {
            Log.d(TAG, "OpenCV failed to load");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_3_0, this, mLoaderCallBack);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        mCameraMat = new Mat(height, width, CvType.CV_8UC4);
    }

    @Override
    public void onCameraViewStopped() {
        mCameraMat.release();

    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        mCameraMat = inputFrame.rgba();

        // Template Matrix requires resizing
        // Because Template Image is bigger resolution that the camera preview
        // Note : NEVER ADD TEMPLATE WITH A RESOLUTION GREATER THAN CAMERA PREVIEW (640x480)
//        Imgproc.resize(mTemplateMat,mTemplateMat,mCameraMat.size());


        /// Create the result matrix
        int result_cols =  mCameraMat.cols() - mTemplateMat.cols() + 1;
        int result_rows = mCameraMat.rows() - mTemplateMat.rows() + 1;
        result.create(result_rows, result_cols, CvType.CV_8UC4);

        /// Do the Matching and Normalize
        int match_method = Imgproc.TM_SQDIFF;
        Imgproc.matchTemplate(mCameraMat, mTemplateMat, result, match_method);

        Core.normalize(result, result, 0, 1, NORM_MINMAX, -1, new Mat());

        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(result);
//
        Point matchLoc;
        /// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
        if( match_method  == Imgproc.TM_SQDIFF || match_method == Imgproc.TM_SQDIFF_NORMED )
        {
            matchLoc = minMaxLocResult.minLoc;
        }
        else
        {
            matchLoc = minMaxLocResult.maxLoc;
        }

        if (minMaxLocResult.maxVal >= threshold) {
            Log.d(TAG, " Best Match in threshold "+ minMaxLocResult.maxVal);
            Imgproc.rectangle(mCameraMat, matchLoc, new Point(matchLoc.x + mTemplateMat.cols(), matchLoc.y + mTemplateMat.rows() ), new Scalar(255,0,0) );

        } else {
            Log.d(TAG, " MatchResult Threshold Value "+ minMaxLocResult.maxVal);

        }

        Imgproc.rectangle(result, matchLoc, new Point(matchLoc.x + mTemplateMat.cols(), matchLoc.y + mTemplateMat.rows()), new Scalar(255, 0, 0));
        return mCameraMat;
    }
}
