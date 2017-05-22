import org.opencv.core.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FastFeatureDetector;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.calib3d.Calib3d.findHomography;
import static org.opencv.features2d.DescriptorExtractor.BRISK;
import static org.opencv.features2d.DescriptorMatcher.BRUTEFORCE;
import static org.opencv.imgcodecs.Imgcodecs.imread;
import static org.opencv.imgcodecs.Imgcodecs.imwrite;
import static org.opencv.imgproc.Imgproc.*;

/**
 * Created by roma on 14.05.17.
 */
public class Main {

    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        imwrite("out.png",
                doo(imread(Main.class.getResource("PNG/output.png").getPath()),
                        imread(Main.class.getResource("PNG/3.png").getPath()), 0.2));


    }

    static Mat doo(Mat image2, Mat image1, double f){
        //        List<Mat> images = new ArrayList<Mat>();
//        for(Integer i = 1; i < 6; i++) {
//            Mat grayImage = new Mat();
//            cvtColor(imread(Main.class.getResource("PNG/" + i + ".png").getPath()), grayImage, );
//            images.add(grayImage.clone());
//        }
        Mat grayImage1 = new Mat();
        Mat grayImage2 = new Mat();
        cvtColor(image1, grayImage1, COLOR_RGB2GRAY);
        cvtColor(image2, grayImage2, COLOR_RGB2GRAY);

//-- Step 1: Detect the keypoints using SURF Detector
        int minHessian = 400;
        FastFeatureDetector detector = FastFeatureDetector.create( 10, true, FastFeatureDetector.TYPE_9_16);
        MatOfKeyPoint keypoints_object = new MatOfKeyPoint();
        MatOfKeyPoint keypoints_scene = new MatOfKeyPoint();

        detector.detect( grayImage1, keypoints_object );
        detector.detect( grayImage2, keypoints_scene );

//-- Step 2: Calculate descriptors (feature vectors)
        DescriptorExtractor extractor = DescriptorExtractor.create(BRISK);

        Mat descriptors_object = new Mat(), descriptors_scene = new Mat();

        extractor.compute( grayImage1, keypoints_object, descriptors_object );
        extractor.compute( grayImage2, keypoints_scene, descriptors_scene );

//-- Step 3: Matching descriptor vectors using FLANN matcher
        DescriptorMatcher matcher = DescriptorMatcher.create(BRUTEFORCE);
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match( descriptors_object, descriptors_scene, matches );

        double max_dist = 0; double min_dist = 100;
//-- Quick calculation of max and min distances between keypoints
        DMatch[] dMatches = matches.toArray();
        for( int i = 0; i < dMatches.length; i++ )
        { double dist = dMatches[i].distance;
            if( dist < min_dist ){ min_dist = dist;}
            if( dist > max_dist ){ max_dist = dist;}
        }

//        printf("-- Max dist : %f \n", max_dist );
//        printf("-- Min dist : %f \n", min_dist );

//-- Use only "good" matches (i.e. whose distance is less than 3*min_dist )
        MatOfDMatch good_matches = new MatOfDMatch();
        for( int i = 0; i < descriptors_object.rows(); i++ )
        {
            if( dMatches[i].distance < max_dist * f){
                good_matches.push_back(new MatOfDMatch(dMatches[i]));
            }
        }
        MatOfPoint2f obj = new MatOfPoint2f();
        MatOfPoint2f scene = new MatOfPoint2f();
        DMatch[] goodDMAtches = good_matches.toArray();
        for( int i = 0; i < goodDMAtches.length; i++ )
        {
            //-- Get the keypoints from the good matches
            obj.push_back(new MatOfPoint2f(keypoints_object.toArray()[ goodDMAtches[i].queryIdx ].pt));
            scene.push_back(new MatOfPoint2f( keypoints_scene.toArray()[ goodDMAtches[i].trainIdx ].pt));
        }

// Find the Homography Matrix
        Mat H = findHomography( obj, scene);
        writeMatToFile("1", H);
        // Use the Homography Matrix to warp the images
        System.out.println();
        Mat result = new Mat();
        warpPerspective(image1,result,H, new Size(image1.cols() + image2.cols(), image1.rows()));
        Mat half = new Mat(result, new Rect(0,0, image2.cols(), image2.rows()));
        List<Point> list = new ArrayList<Point>();
        for(int i = 0; i < result.cols(); i++){
            int a = 0;
            for(int j = 0; j < result.rows(); j++){
                double[] doubles = result.get(j, i);
                if (Arrays.equals(doubles, new double[]{0, 0, 0})){
                    a++;
                }
            }
            if((double)a <= result.rows() * 0.8d){
                for(int j = 0; j < result.rows(); j++){
                    list.add(new Point(j,i));
                }
            }
        }
        image2.copyTo(half);
        MatOfPoint matOfPoint = new MatOfPoint(list.toArray(new Point[list.size()]));
        Rect rect = boundingRect(matOfPoint);
        return  result;
    }
    private static void writeMatToFile(String numberFile, Mat mat){
        try {
            FileOutputStream fileOutputStream = new FileOutputStream(new File("chess_0"+numberFile+".txt"));
            StringBuilder stringBuilder = new StringBuilder();
            int i = 0;
            for(int r = 0; r < mat.rows(); r++){
                for(int c = 0; c < mat.cols(); c++){
                    stringBuilder.append(i++).append(" ").append(Arrays.toString(mat.get(r,c))).append(",");
                }
                if(r != mat.cols() -1)
                    stringBuilder.append("\n");
            }
            fileOutputStream.write(stringBuilder.toString().getBytes());
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
  //  0<= _rowRange.start && _rowRange.start <= _rowRange.end && _rowRange.end <= m.rows
}
