#!/usr/bin/env python3
################################################################################
# This application takes photo's/images from a directory, encode these for face recognition
# and insert these in a .pkl file. Each photo may contain only one person
# and the name of the photo needs to be the name of the person, in the format: firstname lastname,
# optionally followed by -n where n is the sequence number of the photo to allow multiple photo's per person
################################################################################

# imports
import argparse         # to get arguments from commandline
import sys              # to do exit with errorcode
import os               # to open files
import cv2              # to load the open-cv library
import face_recognition # to load the face recognition libary
import pickle           # to load the option to store (image) files
import pathlib          # to get more info of the photo file

# first parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', help='this is the directory to load the know faces')
parser.add_argument('-o', '--output', help='this is the file to store the learned faces')
args = parser.parse_args()
image_dir = args.image
if not image_dir:
    print(f'No directory with known faces specified. Please use: python3 learn_fr.py -i \"~/your input directory" -o \"trained_faces.pkl\"')
    sys.exit(404)  # Bail out with 404 = no known faces directory specified
outputfile = args.output
if not outputfile:
    print(f'No file to store the encoded faces specified. Please use: python3 learn_fr.py -i \"~/your input directory"  -o \"trained_faces.pkl\"')
    sys.exit(404)  # Bail out with 404 = no output file specified

print(f'OpenCV version used: {cv2.__version__}')  # check which version we have for cv2
Encodings = []  # this is the array that hold all the encodings
Names = []  # This is the array that hold all the names
Sequences = []  # this is the sequence of the imagefile
FileDate = []  # this is the date of the imagefile

for root, dirs, files in os.walk(image_dir):
    for file in files:
        fullPath = os.path.join(root, file)
        filename = os.path.splitext(file)[0]
        if ' ' in filename:
            if '-' in filename:
                personname = filename.rsplit('-', 1)[0]  # get name before the -
                sequence = int(filename.rsplit('-', 1)[1])  # get the sequence of the name after the -
            else:
                personname = filename
                sequence = 0
            print(f'Found image of {personname} with sequence {sequence}')
            personimage = face_recognition.load_image_file(fullPath)  # load the image we just found
            encoding = face_recognition.face_encodings(personimage)[0]  # encode the first face in the image (although there is only one)
            Encodings.append(encoding)
            Names.append(personname)
            Sequences.append(sequence)
            fname = pathlib.Path(fullPath)
            FileDate.append(fname.stat().st_ctime)
        else:
            print(f'Ignoring file: {filename}')
print(f'Names found: {Names}')
print(f'With sequences: {Sequences}')
print(f'Created at: {FileDate}')
with open(outputfile, 'wb') as trainedfacesfile:
    # write all learned faces with names in a pickle file
    pickle.dump(Names, trainedfacesfile)
    pickle.dump(Sequences, trainedfacesfile)
    pickle.dump(FileDate, trainedfacesfile)
    pickle.dump(Encodings, trainedfacesfile)
