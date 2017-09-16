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
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.LinkedList;
import java.util.List;

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

    FeatureDetector detector;
    DescriptorExtractor descriptor;
    DescriptorMatcher matcher;
    Mat descriptors2,descriptors1;
    MatOfKeyPoint keypoints1,keypoints2;

    BaseLoaderCallback mLoaderCallBack = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case BaseLoaderCallback.SUCCESS:{

                    detector = FeatureDetector.create(FeatureDetector.ORB);
                    descriptor = DescriptorExtractor.create(DescriptorExtractor.ORB);
                    matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

                    // load the specified image from file system in bgr color
                    Mat bgr = null;
                    try {
                        bgr = Utils.loadResource(getApplicationContext() , R.drawable.template_switchboard2, Imgcodecs.CV_LOAD_IMAGE_COLOR);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // convert the image to rgba
                    mTemplateMat = new Mat();
                    Imgproc.cvtColor(bgr, mTemplateMat, Imgproc.COLOR_BGR2RGBA);//COLOR_BGR2GRAY
                    mTemplateMat.convertTo(mTemplateMat, CvType.CV_8UC1, 255.0/65536.0);
                    descriptors1 = new Mat();
                    keypoints1 = new MatOfKeyPoint();
                    detector.detect(mTemplateMat, keypoints1);
                    descriptor.compute(mTemplateMat, keypoints1, descriptors1);

//                    mCameraMat = new Mat();
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
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

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
        mCameraMat = inputFrame.rgba();//gray()

        // Template Matrix requires resizing
        // Because Template Image is bigger resolution that the camera preview
        // Note : NEVER ADD TEMPLATE WITH A RESOLUTION GREATER THAN CAMERA PREVIEW (640x480)
//        Imgproc.resize(mTemplateMat,mTemplateMat,mCameraMat.size());


        descriptors2 = new Mat();
        keypoints2 = new MatOfKeyPoint();
        detector.detect(mCameraMat, keypoints2);
        descriptor.compute(mCameraMat, keypoints2, descriptors2);

        // Matching
        MatOfDMatch matches = new MatOfDMatch();
        if (mTemplateMat.type() == mCameraMat.type()) {
            matcher.match(descriptors1, descriptors2, matches);
        } else {
            return mCameraMat;
        }

        Double max_dist = 0.0;
        Double min_dist = 100.0;
        List<DMatch> matchesList = matches.toList();

        for (int i = 0; i < matchesList.size(); i++) {
            Double dist = (double) matchesList.get(i).distance;
            if (dist < min_dist)
                min_dist = dist;
            if (dist > max_dist)
                max_dist = dist;
        }



        /// Create the result matrix
        int result_cols =  mCameraMat.cols() - mTemplateMat.cols() + 1;
        int result_rows = mCameraMat.rows() - mTemplateMat.rows() + 1;
        result.create(result_rows, result_cols, CvType.CV_32FC1 );

        /// Do the Matching and Normalize
        // TM_CCOEFF 3.21 FPS [Not so accurate but not bad too]
        // TM_CCORR 3.78 FPS very in-accurate results
        // TM_SQDIFF 2.93 FPS Much accurate  Remember to set the threshold to lower value for this method
        int match_method = Imgproc.TM_CCOEFF;
        Imgproc.matchTemplate(mCameraMat, mTemplateMat, result, match_method);

        Core.normalize(result, result, 0, 1, NORM_MINMAX, -1, new Mat());

        Core.MinMaxLocResult minMaxLocResult = Core.minMaxLoc(result, new Mat());

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
            // Red LandMark Scalar(255, 0, 0) )
            // Imgproc.rectangle draws landmark around matching area
//        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
            for (int i = 0; i < matchesList.size(); i++) {
//                good_matches.addLast(matchesList.get(i));
                if (matchesList.get(i).distance <= (1.5 * min_dist)) {
                    Log.d(TAG, " Best Match in threshold "+ minMaxLocResult.maxVal);
                    Imgproc.rectangle(mCameraMat, matchLoc, new Point(matchLoc.x + mTemplateMat.cols(), matchLoc.y + mTemplateMat.rows() ), new Scalar(0, 255, 0) );
                }
            }

        } else {
            Log.d(TAG, " MatchResult Threshold Value "+ minMaxLocResult.maxVal);

        }

        Imgproc.rectangle(result, matchLoc, new Point(matchLoc.x + mTemplateMat.cols(), matchLoc.y + mTemplateMat.rows()), new Scalar(0, 255, 0));
        return mCameraMat;
    }
}
