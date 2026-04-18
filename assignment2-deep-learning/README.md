Overview of the Pipeline
1. Image acquisition

The first stage is capturing images from a webcam using OpenCV. When the camera is started, the system continuously reads video frames and stores the latest frame for further processing. This live stream acts as the input to both enrollment and recognition.

2. Face detection and alignment

Each frame is converted from OpenCV’s BGR format to RGB format before being passed to the MTCNN detector. MTCNN is responsible for locating faces in the frame and, in your implementation, also producing aligned face crops of size 160 × 160. During recognition, the multi-face version of MTCNN is used so that more than one face can be detected at once. During enrollment, the single-face version is used so that only one clear face is captured for a person being registered.

3. Preprocessing

Preprocessing is partly handled explicitly and partly handled inside MTCNN. Explicitly, the frame is converted from BGR to RGB. Internally, MTCNN performs face cropping, adds the specified margin, resizes the face to the required input size, and returns a tensor ready for the recognition model. This means the detected face is standardized before embedding extraction.

4. Embedding extraction

Once an aligned face tensor is obtained, it is passed into the InceptionResnetV1 model pretrained on VGGFace2. This model converts the face image into a 512-dimensional feature vector, commonly called a face embedding. The purpose of the embedding is to represent the identity of a face numerically so that faces of the same person are close together in feature space, while faces of different people are farther apart.

5. Database creation and storage

During enrollment, the system captures multiple samples of a person from the webcam. For each accepted sample, the original image is saved into the dataset folder and its embedding is generated immediately. These embeddings are stored in .npy files together with numeric labels, while a JSON class map links each label to a person’s name. This creates the known-face database used later during recognition.

6. Matching and identity decision

In recognition mode, embeddings from live faces are compared against all stored embeddings in the database using Euclidean distance. The smallest distance is selected as the best match. If this minimum distance is below a user-defined threshold, the system assigns the corresponding person’s name. Otherwise, the face is labeled as Unknown. This threshold-based comparison allows the system to distinguish between known and unseen faces.

7. Result visualization

After matching, the system draws a bounding box around each detected face and displays the predicted name on the live video feed. If the face is not recognized, the label shown is “Unknown.” In enrollment mode, the interface instead shows readiness feedback and sample capture progress. A Tkinter GUI provides controls for starting the camera, reloading the database, preparing enrollment, and adjusting recognition parameters such as the threshold and recognition frequency.
