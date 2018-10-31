# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view")
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeup_images/undistorted.png "Undistorted"
[image2]: ./writeup_images/undistorted_real.png "Road Transformed"
[image3]: ./writeup_images/binary.png "Binary Image"
[image4]:  ./writeup_images/bird_eye.png "Warp Image"
[image5]: ./writeup_images/polyfit.png "Fit Image"
[image6]: ./writeup_images/lane_drawn.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

#### Camera Calibration

##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image

The camera calibration is done with the help of the chessboard images, taken from the same camera on different angles. Using the `cv2.findChessboardCorners` function and given the number of corners which are `(9,6)`, the object and image points are calculated. With the image points and the object points are used with the `cv2.calibrateCamera`. This results the camera and distortion matrix. As it is the same for all the camera images, it is saved in the pickle file so it can used again to undistort.

```python
def extract_image_and_object_points(image_dir, chessboard_x=9, chessboard_y=6):
    """
    Extracts the image points and object points from the given image_dir

    Args:
        image_dir:    The path glob which can be used to find the images
        chessboard_x: The chessboard corners in the x axis
        chessboard_y: The chessboard corners in the y axis

    Returns:
        The pickle file that contain both of these data.
    """
    objp = np.zeros((chessboard_x * chessboard_y, 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_x, 0:chessboard_y].T.reshape(-1, 2)
    objpoints = []
    imgpoints = []
    images = glob.glob(image_dir)
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, (chessboard_x, chessboard_y), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return imgpoints, objpoints

def get_image_and_distortion_matrix(image_dir,
                                    test_image,
                                    chessboard_x=9,
                                    chessboard_y=6):
    """
    Returns the image matrix and distortion matrix for the given image, image points and object points

    Args:
        image_dir:    The path glob which can be used to find the images
        test_image:   The array like image or PIL image
        chessboard_x: The chessboard corners in the x axis
        chessboard_y: The chessboard corners in the y axis

    Returns:
        The pickel file that contains the distortions
    """
    imgpoints, objpoints = extract_image_and_object_points(image_dir)
    img_size = (test_image.shape[1], test_image.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img_size, None, None)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    dist_pickle['imgpoints'] = imgpoints
    dist_pickle['objpoints'] = objpoints
    pickle.dump(dist_pickle, open("calibrate_dist_pickle.p", "wb"))


def undistort_image(img, calibrate_pickle='calibrate_dist_pickle.p'):
    """
    Undistorts the given image with the calibration pickel

    Args:
        img: The image that should be used is the array like image or PIL
        calibrate_pickle: The pickle file that should be used for the source of calibration

    Returns:
        The undistorted image
    """
    dist_pickle = pickle.load(open(calibrate_pickle, "rb"))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist
```

![undistorted image][image1]
*On the left is the distorted and on the right is the undistorted image.*

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The same method above is used for the road image. For all the test images, this step is done, and they results can be found in `output_images/undistorted_*`. An Example can be found here:
![undistorted real image][image2]
*On the left is the distorted image and on the right is the undistorted image.*

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Using the combination of color transform and Sobel function (Gradient) and thresholds it is tried to isolate the lane lines they can be detected by other algorithm afterward.
For all the test images this transformation is done and they can be found by `output_images\`

```python
def hls_select_s(img, thresh=(0, 255)):
    """
    Applies the HLS threshold on the given image.

    Args:
        img: The image that should be used
        thresh: The threshold for the given image

    Returns:
        img
    """
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = img_hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary

def sobel_process(img, orient='x', thresh_min=20, thresh_max=100):
    """
    Adds the sobel process to the given image.

    Args:
        img: The image like array
        oreint: The Sobel orientation, could be x or y
        thresh_min: The minimum threshold for the sobel
        thresh_max: The maximum threshold for the sobel

    Return:
        image like array that is processed.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = cv2.normalize(abs_sobel, None, 0.0, 255.0, cv2.NORM_MINMAX,
                                 cv2.CV_64F)
    sxbinary = cv2.inRange(scaled_sobel, thresh_min, thresh_max)
    retval, sxbinary = cv2.threshold(sxbinary, 250, 1.0, cv2.THRESH_BINARY)
    return sxbinary

def get_binary_image(image,
                     mag_thresh=(0, 255),
                     hls_thresh=(0, 255)):
    """
    Creates the binary image from the given image with help of Sobel and HLS colour space

    Args:

        image: The image like array or the PIL image
        mag_thresh: The threshold that should be used for the Sobel magnitude
        hls_thresh: The threshold that should be used for the hls colour limiting

    Returns:
        The filtered binary image
    """
    hsl_image = hls_select_s(image, hls_thresh)
    threshold_image = sobel_process(image, 'x', mag_thresh[0], mag_thresh[1])
    combined_binary = np.zeros_like(threshold_image)
    combined_binary[(hsl_image == 1) | (threshold_image == 1)] = 1
    return combined_binary

img = mpimg.imread(path)
image_name = os.path.basename(path)
img = undistort_image(img)
binary = get_binary_image(img, (20,100),(170,255))
```

The thresholds used for the HLS, s binary image are `(170,255)` and for sobel is `(20, 100)`. All the test images are binary transformed using the color space and sobel and can be found by `output_images/binary_*`.

![Binary image][image3]
*On the left is the normal image, and on the right is the binary transformed image.*

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For the perspective transform the region of the interest is used as:

```python
    imshape = img.shape

    left_point =  (1.5 * imshape[1]/8, imshape[0]- 50)
    source_apex1 = (imshape[1]/2 - 50 , imshape[0]/2 + 85 )
    source_apex2 = (imshape[1]/2 + 50, imshape[0]/2 + 85 )
    right_point = (6.5 * imshape[1]/8, imshape[0]- 50)
    source = np.float32([[left_point, source_apex1, source_apex2, right_point]])

    dest_left_point =  (1.5 * imshape[1]/8  + offset, imshape[0])
    dest_apex1 = (1.5 * imshape[1]/8 + offset, 0)
    dest_apex2 = (6.5 * imshape[1]/8 - offset, 0)
    dest_right_point = (6.5 * imshape[1]/8 - offset, imshape[0])
    dest = np.float32([[dest_left_point, dest_apex1, dest_apex2, dest_right_point]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 240, 670      | 290, 720    |
| 590, 445      | 290, 0      |
| 1040, 670     | 990, 720    |
| 690, 445      | 990, 0      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. With this the car part in the front is also removed from the image.

```python
def transform_bird_eye(img, offset = 50):
    """
    Transform the ROI (Region of Interest) of the given image to bird eye

    Args:
        img: The array like image or PIL image
        offset: The offset for the transformation
    Return:
            The transformed image
            The transform matrix
            The inverse of transform matrix
    """
    imshape = img.shape
    img_size = (imshape[1], imshape[0])
    left_point =  (1.5 * imshape[1]/8, imshape[0]- 50)
    source_apex1 = (imshape[1]/2 - 50 , imshape[0]/2 + 85 )
    source_apex2 = (imshape[1]/2 + 50, imshape[0]/2 + 85 )
    right_point = (6.5 * imshape[1]/8, imshape[0]- 50)
    source = np.float32([[left_point, source_apex1, source_apex2, right_point]])
    dest_left_point =  (1.5 * imshape[1]/8  + offset, imshape[0])
    dest_apex1 = (1.5 * imshape[1]/8 + offset, 0)
    dest_apex2 = (6.5 * imshape[1]/8 - offset, 0)
    dest_right_point = (6.5 * imshape[1]/8 - offset, imshape[0])
    dest = np.float32([[dest_left_point, dest_apex1, dest_apex2, dest_right_point]])
    M = cv2.getPerspectiveTransform(source, dest)
    Minv = cv2.getPerspectiveTransform(dest, source)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M, Minv
```

The bird eve images of the test images can be found by `output_images\bird_eye*`.
An example for the warped image can be found

![alt text][image4]
*The basic image can be found on the left and transformed image is on the right*

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial

With all the steps described above in order and the sliding window method, the lane pixels for the left and right lane are found and the with `np.polyfit()`function the polynom representing them is found.

* undistort image
* binary transform
* perspective transform
* sliding window and polyfit

```python
def find_lane_pixels(binary_warped):
    """
    Finds the lane pixels in a binary warped image

    Args:
        binary_warped: The binary warped array_like image that should be used to find line

    Returns:
        The arrays containing points for the left and right image
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped, draw_poly = True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[
            1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty**2 + 1 * ploty
        right_fitx = 1 * ploty**2 + 1 * ploty
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
    if draw_poly is True:
        left_points = np.array(list(zip(left_fitx, ploty)),  np.int32)
        right_points = np.array(list(zip(right_fitx, ploty)),  np.int32)
        cv2.polylines(out_img, [left_points], False, (255,255,0),2)
        cv2.polylines(out_img, [right_points], False, (255,255,0), 2)
    return left_fit, right_fit, left_fitx, right_fitx, ploty, out_img
```

The transformed image can be found by `output_images\lane_found_*`. An example of the transformed image can be seen next.

![alt text][image5]
*on the left the basic image and on the right is the fit image with the polynom drawn*

This takes a lot of time, if the the line is found in the frames before it can still be used in the given binary image to find the existing line. This is done with the help of `simple_fit` function in the code.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center

For the curvature calculation the given formula is used. For the whole frame the average of the curvature of the left and right lane is used. For the position of the car from center, the distance between the left and right lane is calculated and then this is subtracted from the center of the image. The absolute value of this will be the position with respect to center. The image pixel to real world conversion should also be done at some point in this process.

```python
def generate_data(leftx, rightx, ploty, ym_per_pix, xm_per_pix):
    '''
    Generates fake data to use for calculating lane curvature.
    In your own project, you'll ignore this function and instead
    feed in the output of your lane detection algorithm to
    the lane curvature calculation.
    '''
    dest = abs(leftx[-1] - rightx[-1])*xm_per_pix
    center_of_image = abs(xm_per_pix * 660)
    dest_from_center = abs(dest - center_of_image)
    left_fit_cr  = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix , 2)
    right_fit_cr = np.polyfit(ploty* ym_per_pix, rightx * xm_per_pix, 2)
    return ploty, left_fit_cr, right_fit_cr, dest_from_center

def measure_curvature_real(left_fit, right_fit, ploty):
    '''
    Calculates the curvature of polynomial functions in meters.

    Args:
        left_fit: The line that is fitted to the left lane
        right_fit: The line that is fitted tot the right lane
        ploty: The ploty image.

    Returns:
        left_curverad, right_curverad
    '''
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    ploty, left_fit_cr, right_fit_cr, dest_from_center = generate_data(left_fit, right_fit, ploty, ym_per_pix, xm_per_pix)
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)** (3/2))/ np.absolute((2 * left_fit_cr[0]))
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix  + right_fit_cr[1])**2)** (3/2))/ np.absolute((2 * right_fit_cr[0]))
    return left_curverad, right_curverad, dest_from_center
```

For the test2 image, the calculate values are: `Curvature: 337.34735113522447 360.93948603163614, Dest: 0.08362889746707713`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

With the draw_lane_boundry and inverse transformation matrix the found lane will be marked on the image.

```python
def draw_lane_boundires(img, sobel_thresh = (20, 150), hls_thresh=(150, 255)):
    """
    Draw lane boundries on the image

    Args:
        img: The img array
        sobel_thresh: The sobel threshold
        hls_thresh: The threshold for the hls image filter

    Return:
        img
    """
    undist = undistort_image(img)
    top_down = get_binary_image(undist, sobel_thresh, hls_thresh)
    warped, M, Minv = transform_bird_eye(top_down)
    left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = fit_polynomial(warped)
    left_curverad, right_curverad, dest_from_center =  measure_curvature_real(left_fitx, right_fitx, ploty)
    curvature = sanitize_curvature(left_curverad, right_curverad)
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    curvature_text = "Curvature:% 6.0fm" % curvature
    position_text = "Position: % 5.2fm" % dest_from_center
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, curvature_text,(100,70), font, 2,(0,0,0),2)
    cv2.putText(result, position_text, (150,120), font, 2,(0,0,0),2)
    return result
```

The transformed test images can be found by `output_images/lane_drawn_`. For the example image the result can be found below:

![alt text][image6]
*on the left the basic image and on the right is the fit image with lane drawn*

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!)

For the pipeline the following code is used. It smooths the line with using the average of last 10 framses, it also does sanity check for the calculated lane lines on the given image.

```python
MAX_ACCEPTABLE_WIDTH = 1.0 # in meters
MAX_ACCEPTABLE_PARALEL = 5.0 # in percent
MAX_ACCEPTABLE_LINE_MOVED = .2 # in meters

# Define a class to receive the characteristics of each line detection
class Line(object):
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line for the left lane
        self.recent_xfitted_left = deque(maxlen = 10)
        # x values of the last n fits of the line for the right lane
        self.recent_xfitted_right = deque(maxlen =10)
        # average x values of the fitted line over the last n iterations for the left lane
        self.best_x_left = None
        # average x values of the fitted line over the last n iteration for the right lane
        self.best_x_right = None
        # polynomial coefficients averaged over the last n iterations for the left lane
        self.best_fit_left = None 
        # polynomial coefficients averaged over the last n iterations for the right lane
        self.best_fit_right = None
        # polynomial coefficients for the last n fits for the left lane
        self.recent_fits_left = deque(maxlen =10)
        # polynomial coefficients for the last n fits for the right lane
        self.recent_fits_right = deque(maxlen =10)
        # polynomial coefficients for the most recent fit for the left lane
        self.current_fit_left = [np.array([False])] 
        # polynomial coeficients for the most recent fir for the right lane
        self.current_fit_right = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits for the left lane
        self.diffs_left = np.array([0,0,0], dtype='float')
        #difference in fit coefficients between last and new fits for the right lane
        self.diffs_right = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels for left lane
        self.allx_left = None 
        #x values for detected line pixels for left right
        self.allx_right = None
        #y values for detected line pixels
        self.ally = None

    def get_default_frame_data(self):
        """
        Returns the default frame_data for the given time ine the line

        Returns:
            frame_data: The FrameData object that should be used.
        """
        frame_data = FrameData()
        frame_data.left_fit = self.best_fit_left
        frame_data.right_fit = self.best_fit_right
        frame_data.left_fitx = self.best_x_left
        frame_data.right_fitx = self.best_x_right
        frame_data.ploty = ploty
        frame_data.curvature = self.radius_of_curvature
        frame_data.dest_from_center = self.line_base_pos
        return frame_data

class FrameData(object):
    """
    The data that is extracted form the given frame.
    """
    def __init__(self):
        self.left_fit = None
        self.left_fitx = None 
        self.right_fit = None 
        self.right_fitx = None
        self.ploty = None 
        self.curvature = None
        self.right_curverad = None
        self.left_curverad = None
        self.dest_from_center = None

def fit_poly(img_shape, leftx, lefty, rightx, righty):
     ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    return left_fit, right_fit, left_fitx, right_fitx, ploty

def brute_force_fit(img, sobel_thresh = (20, 150), hls_thresh=(150, 255)):
    """
    This is the brute force fit that is used only when the normal look a head fitting does not work.

    Args:
        img:          The array like image or PIL image
        sobel_thresh: The size of the sobel threshold
        hls_thresh:   The threshold for the hls binary

    Returns:
        frame_data: The frame data which contains all the frame related data.
        M:          The Matrix used to the transform the image to bird_eye view
        Minv:       The inverse of the matrix that is used to transform images to bird_eye view
    """
    img = undistort_image(img)
    top_down = get_binary_image(img, sobel_thresh, hls_thresh)
    top_down, M, Minv = transform_bird_eye(top_down)
    left_fit, right_fit, left_fitx, right_fitx, ploty, out_img = fit_polynomial(top_down)
    left_curverad, right_curverad, dest_from_center =  measure_curvature_real(left_fitx, right_fitx, ploty)
    frame_data = FrameData()
    frame_data.left_fit = left_fit
    frame_data.right_fit = right_fit
    frame_data.left_fitx = left_fitx
    frame_data.right_fitx = right_fitx
    frame_data.ploty = ploty
    frame_data.right_curverad = right_curverad
    frame_data.left_curverad = left_curverad
    frame_data.curvature = sanitize_curvature(left_curverad, right_curverad)
    frame_data.dest_from_center = dest_from_center
    return frame_data, M, Minv

def simple_fit(img, line,  sobel_thresh= (20, 150), hls_thresh=(150, 255)):
    """
    This is is the simple fit which does not used the sliding window approach to find the lane lines

    Args:
        img:  The array like image or PIL image
        line: The line object which contains the history of the operation

    Returns:
        frame_data: The frame data that can be used for the sanity_check or being added to the list of lines
    """
    left_fit = line.current_fit_left
    right_fit = line.current_fit_right
    binary_image = get_binary_image(img, sobel_thresh, hls_thresh)
    binary_warped, M, Minv = transform_bird_eye(binary_image)
    margin = 100
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit_poly, right_fit_poly, left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    left_curverad, right_curverad, dest_from_center =  measure_curvature_real(left_fitx, right_fitx, ploty)
    frame_data = FrameData()
    frame_data.left_fit = left_fit_poly
    frame_data.right_fit = right_fit_poly
    frame_data.left_fitx = left_fitx
    frame_data.right_fitx = right_fitx
    frame_data.ploty = ploty
    frame_data.right_curverad = right_curverad
    frame_data.left_curverad = left_curverad
    frame_data.curvature = sanitize_curvature(left_curverad, right_curverad)
    frame_data.dest_from_center = dest_from_center
    return frame_data

def sanitiy_check(frame_data, line):
    """
    Checks if the given line is a valid line with the sanity check.

    Args:
        image: The image that should be used.
        line: The line object that should be used for the line

    Returns:
        bool
    """
    xm_per_pix = 3.7/700
    lane_width = abs(frame_data.left_fitx[-1] - frame_data.right_fitx[-1]) * xm_per_pix
    if np.abs(lane_width - 3.7) > MAX_ACCEPTABLE_WIDTH:
        return False
    curve_differs = np.abs((frame_data.left_curverad - frame_data.right_curverad) / frame_data.left_curverad)
    if curve_differs > MAX_ACCEPTABLE_PARALEL:
        return False
    lane_moved = np.abs(frame_data.dest_from_center - line.line_base_pos)
    if lane_moved > MAX_ACCEPTABLE_LINE_MOVED:
        return False
    return True

def add_line(line, frame_data):
    """
    Adds a new line to the given buffer.

    Args:
        line: The empty line object which will be field with the data of the
        frame_data: The data that is extracted from the given frame

    Returns:
        line: The mainpulate line buffer
    """
    if len(line.recent_xfitted_left) == 0:
        ## If the buffer is empty fill it with the first value.
        line.detected = True
        line.recent_xfitted_left.append(frame_data.left_fitx)
        line.recent_xfitted_right.append(frame_data.right_fitx)
        line.best_x_left = frame_data.left_fitx
        line.best_x_right = frame_data.right_fitx
        line.best_fit_left = frame_data.left_fit
        line.best_fit_right = frame_data.right_fit
        line.recent_fits_left.append(frame_data.left_fit)
        line.recent_fits_right.append(frame_data.right_fit)
        line.current_fit_left = frame_data.left_fit
        line.current_fit_right = frame_data.right_fit
        line.radius_of_curvature = frame_data.curvature
        line.line_base_pos = frame_data.dest_from_center
        line.allx_left = frame_data.left_fitx
        line.allx_right = frame_data.right_fitx
        line.ally = frame_data.ploty
    else:
        line.detected = True
        line.recent_xfitted_left.append(frame_data.left_fitx)
        line.recent_xfitted_right.append(frame_data.right_fitx)
        line.best_x_left = np.mean(np.array(line.recent_xfitted_left),axis=0)
        line.best_x_right = np.mean(np.array(line.recent_xfitted_right),axis=0)
        line.best_fit_left = np.mean(np.array(line.recent_fits_left),axis=0)
        line.best_fit_right = np.mean(np.array(line.recent_fits_right),axis=0)
        line.recent_fits_left.append(frame_data.left_fit)
        line.recent_fits_right.append(frame_data.right_fit)
        line.diffs_left = np.absolute(line.current_fit_left - frame_data.left_fit)
        line.diffs_right = np.absolute(line.current_fit_right - frame_data.right_fit)
        line.current_fit_left = frame_data.left_fit
        line.current_fit_right =frame_data.right_fit
        line.radius_of_curvature = frame_data.curvature
        line.line_base_pos = frame_data.dest_from_center
        line.allx_left = frame_data.left_fitx
        line.allx_right = frame_data.right_fitx
        line.ally = frame_data.ploty
    return line

def draw_frame(img, frame_data):
    """
    Draws the frame_data retrieved data on the given image.

    Args:
        img:        The array like image or image file
        frame_data: The frame_data calculated data

    Return:
        img
    """
    undist = undistort_image(img)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    top_down, M, Minv = transform_bird_eye(gray)
    warp_zero = np.zeros_like(top_down).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    left_fitx = frame_data.left_fitx
    right_fitx = frame_data.right_fitx
    ploty = frame_data.ploty
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    curvature_text = "Curvature:% 6.0fm" % frame_data.curvature
    position_text = "Position: % 5.2fm" % frame_data.dest_from_center
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, curvature_text,(100,70), font, 2,(0,0,0),2)
    cv2.putText(result, position_text, (150,120), font, 2,(0,0,0),2)
    return result

def pipeline(image, **params):
    """
    The pipeline for the image processing

    Args:
        image: The array like image or PIL image
        params: The parameters that should be used.

    Returns: The finished image.
    """
    line = params["line"]
    if len(line.recent_xfitted_left) == 0:
        frame_data, M, Minv = brute_force_fit(image)
        add_line(line, frame_data)
    else:
        frame_data = simple_fit(image, line)
        if sanitiy_check(frame_data, line):
            add_line(line, frame_data)
    frame_data = line.get_default_frame_data()
    return draw_frame(image, frame_data)
```

The pipeline video can be found in the file [`project_video_output.mp4`](./project_video_output.mp4).

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

 1. The pipeline will probably fail on the road with small contrast between the lane and road or the objects beside the road.
 2. When the first images that are used to create the warped binary or not good, or hard the extract information from, the pipeline will fail to
 correct it self.
 3. The pipeline does not recalculate the lane positions using the sliding window (or convolution) when it does not found any lanes using
 simple marginal fitting.
 4. On the roads that have a lot of curves the pipeline does not correct it self.
 5. Smoothing of the frames can be done much better.