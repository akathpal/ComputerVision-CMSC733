To run the program:

Download and Copy the weights file from PRNet-

https://drive.google.com/file/d/1UoE-XuW1SDLUjZmJPkIZ1MLxvQFgmTFH/view

copy to prnet/Data/net-data

Run the code and pass the desired parameters - 

python Wrapper.py

--mode, default=1 
1- swap face in video with image, 
2- swap two faces within video'

--input_path = default="./TestSet/" - directory path
--face default=Rambo - 'path to face'
--video default=Test1 - 'path to the input video'
--method default=tps, 'affine, tri, tps, prnet')
--resize default=False, 'True or False input resizing')


prnet - module consisting of modified prnet files to run the network
traditional - module consisting of all the traditional approach files

Wrapper.py - final python file importing from both of the above modules.

Data - 
Outputs from my own collected data

OutputTestSet-
Outputs from test dataset
