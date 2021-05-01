# NN-SEMG
A neural network designed to classify muscle movement based on SEMG readings from SEMG sensor

Source:
Kaggle/Kirill Yashuk/Classify gestures by reading muscle activity.

Description:
Four classes of motion were written from MYO armband. The MYO armband has 8 sensors placed on the skin surface, each measuring electrical activity produced by muscles beneath.
Each dataset line has 8 consecutive readings of all 8 sensors. So, 64 columns of EMG data. The first 8 columns are the readings of the 8 individual sensors and in this way every 8 columns represent the reading of the 8 sensors.The last column is a resulting gesture that was made while recording the data (classes 0-3)

Data was recorded at 200 Hz, which means that each line is 8*(1/200) seconds = 40ms of record time.
A classifier given 64 numbers would predict a gesture class (0-3).
Gesture classes were: rock - 0, scissors - 1, paper - 2, ok - 3. Rock, paper, scissors gestures are like in the game with the same name, and the OK sign is an index finger touching the thumb and the rest of the fingers spread. Gestures were selected pretty much randomly.

Rock Gesture - 0
Scissor Gesture -1
Paper Gesture - 2
OK Gesture - 3


Each gesture was recorded 6 times for 20 seconds. Each time recording started with the gesture being already prepared and held. Recording stopped while the gesture was still being held. In total there are 120 seconds of each gesture being held in fixed position. All of them recorded from the same right forearm in a short time span. Every recording of a certain gesture class was concatenated into a .csv file with a corresponding name (0-3).[3]
The original dataset consisted of 4 separate csv files where each csv file consisted of the readings of only 1 gesture. For the purpose of this research, the four files were merged and the readings in the final csv file was randomised in order for better training of the neural network.

Algorithm Used:
Feed Forward Neural Network
