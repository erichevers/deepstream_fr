#!/usr/bin/env python3
################################################################################
# This application is based on the NVIDIA Deepstream SDK demo apps
# located at: https://github.com/NVIDIA-AI-IOT/deepstream_python_apps
# In particular the deepstream-imagedata-multistream is taken as the basis
################################################################################

# start with the imports
import sys
sys.path.append('../')
import gi
import configparser
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from gi.repository import GLib
from ctypes import *
import time
import math
import platform
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS
import numpy as np
import pyds
import cv2
import os
import os.path
from os import path
import face_recognition  # to load the face recognition libary
import pickle            # to load the trained faces file
from pathlib import Path  # to auto create directories
from GenericFunctions import init_log

# initiate deepstream variables
fps_streams = {}
frame_count = {}
saved_count = {}
global PGIE_CLASS_ID_VEHICLE
PGIE_CLASS_ID_VEHICLE = 0
global PGIE_CLASS_ID_PERSON
PGIE_CLASS_ID_PERSON = 2
MAX_DISPLAY_LEN = 64
# PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_BICYCLE = 1
# PGIE_CLASS_ID_PERSON = 2
PGIE_CLASS_ID_ROADSIGN = 3
MUXER_OUTPUT_WIDTH = 1920
MUXER_OUTPUT_HEIGHT = 1080
MUXER_BATCH_TIMEOUT_USEC = 4000000
TILED_OUTPUT_WIDTH = 1920
TILED_OUTPUT_HEIGHT = 1080
GST_CAPS_FEATURES_NVMM = "memory:NVMM"
pgie_classes_str = ["Vehicle", "TwoWheeler", "Person", "RoadSign"]

# logging variables
process = 'Deepstream_fr'
logdir = 'logdir/'
logfile = logdir + process + '.log'
loglevel = 'Info'
logsize = 200000
logbackups = 5
global log
log = None

# other variables
learnedfile = logdir + 'known_faces/trained_faces.pkl'
unknow_face_dir = logdir + 'unknown_faces/'
gui = True  # use the GUI as output to show the stream
save_unknown = True  # save unknow faces
unknown_face_name = 'Unknown'  # what name to use when the face is not recognized
sampling_rate = 20   # process every X frames in a stream i.e. sampling_rate of 2 will process every other frame, 4 will 1 frame of 4 etc
resize_factor = 2   # to resize the video stream, set to 4 for high quality streams (> 1920x1080:24fps) to shrink the resolution for better performance
up_scale = 2     # This finds faces better when they are small (1 = standard, 3 & 4 is slow)
detection_model = 'hog'  # this is the model that is used for face recognition: models can be "hog", "cnn". cnn is more accurate, but takes longer
number_jitters = 1             # How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
encoding_model = 'large'    # this is to define the number of landmarks to use 5 (small) or all (large)
capture_filename = 'Frame-'  # this is the name to be used when a screenshot is made of the image
unknown_face_filename = 'UnknownFace-'  # this is the name to be used when an unknow face is found
border = 50  # store unknown face with some surrounding
transcoder = cv2.CAP_FFMPEG  # cv2.CAP_V4L cv2.CAP_V4L2 cv2.CAP_DC1394 cv2.CAP_GSTREAMER or cv2.CAP_FFMPEG
# additional global variables
global Names
Names = []
global Sequence
Sequence = []
global Encodings
Encodings = []
global folder_name
folder_name = None


# tiler_sink_pad_buffer_probe  will extract metadata received on tiler src pad
# and update params for drawing rectangle, object information etc.
def tiler_sink_pad_buffer_probe(pad, info, u_data):
    frame_number = 0
    num_rects = 0
    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
            frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number = frame_meta.frame_num
        l_obj = frame_meta.obj_meta_list
        num_rects = frame_meta.num_obj_meta
        is_first_obj = True
        save_image = False
        obj_counter = {
            PGIE_CLASS_ID_VEHICLE: 0,
            PGIE_CLASS_ID_PERSON: 0,
            PGIE_CLASS_ID_BICYCLE: 0,
            PGIE_CLASS_ID_ROADSIGN: 0
        }
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            # Periodically check for objects with borderline confidence value that may be false positive detections.
            # If such detections are found, annoate the frame with bboxes and confidence value.
            # Save the annotated frame to file.
            if((saved_count["stream_" + str(frame_meta.pad_index)] % 30 == 0) and (obj_meta.confidence > 0.3 and obj_meta.confidence < 0.31)):
                if is_first_obj:
                    is_first_obj = False
                    # Getting Image data using nvbufsurface
                    # the input should be address of buffer and batch_id
                    n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                    # convert python array into numpy array format.
                    frame_image = np.array(n_frame, copy=True, order='C')
                    # covert the array into cv2 default color format
                    frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGRA)

                save_image = True
                frame_image = draw_bounding_boxes(frame_image, obj_meta, obj_meta.confidence)
            # save and draw box when we have a person
            if obj_meta.class_id == PGIE_CLASS_ID_PERSON and frame_meta.pad_index % sampling_rate == 0:
                # only do face recognition if the object is a person and once in a while depending on the sampling rate
                # Getting Image data using nvbufsurface
                # the input should be address of buffer and batch_id
                n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)
                # convert python array into numpy array format.
                frame_image = np.array(n_frame, copy=True, order='C')
                # it looks like the image coming from the buffer includes alpha channel and is in RGB,
                # face recognition also uses RGB, but may be not with alpha channel, so let's remove this to be sure and remove next line if alpha channel is OK
                frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2RGB)
                frame_image = draw_bounding_boxes(frame_image, obj_meta, obj_meta.confidence)
            # continue with the next object when there is one
            try:
                l_obj = l_obj.next
            except StopIteration:
                break

        print("Frame Number=", frame_number, "Number of Objects=", num_rects, "Vehicle_count=", obj_counter[PGIE_CLASS_ID_VEHICLE], "Person_count=", obj_counter[PGIE_CLASS_ID_PERSON])
        # Get frame rate through this probe
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        if save_image:
            cv2.imwrite(f'{folder_name}/stream_{frame_meta.pad_index}/frame_{frame_number}.jpg', frame_image)
        saved_count["stream_" + str(frame_meta.pad_index)] += 1
        try:
            l_frame = l_frame.next
        except StopIteration:
            break
    return Gst.PadProbeReturn.OK


def face_recog(image, obj_meta):
    # Find all the faces and face encodings in the current frame of video
    rgb_small_frame = image
    frame = image
    face_locations = face_recognition.face_locations(rgb_small_frame, number_of_times_to_upsample=up_scale, model=detection_model)
    if face_locations:
        log.info(f'--- Number of faces found in image: {len(face_locations)}')
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, num_jitters=number_jitters, model=encoding_model)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(Encodings, face_encoding)
        name = unknown_face_name
        # use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(Encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = f'{Names[best_match_index]}-{Sequence[best_match_index]}'
        face_names.append(name)
        # some logging below
        if name == unknown_face_name:
            log.info('-- Unknown face detected')
        else:
            log.info(f'-- Known face detected: {name}')

    # display the results
    cvfont = cv2.FONT_HERSHEY_SIMPLEX
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # not required here anymore, because the frame is already in the cv2 format
    for (top, right, bottom, left), face_name in zip(face_locations, face_names):
        if resize_factor != 1:
            top *= resize_factor
            right *= resize_factor
            bottom *= resize_factor
            left *= resize_factor
        # save the unknown faces and do this before the rectangle is inserted in the frame
        if face_name == unknown_face_name and save_unknown:
            j = 0
            while os.path.exists(f'{unknow_face_dir}{unknown_face_filename}{j}.jpg'):
                j += 1
            cv2.imwrite(f'{unknow_face_dir}{unknown_face_filename}{j}.jpg', frame[top - border:bottom + border, left - border:right + border])  # only save the face with a border -> crop image using numphy
        # draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        # draw a box above the face
        cv2.rectangle(frame, (left, top - 25), (left + 200, top), (0, 255, 255), -1)
        cv2.putText(frame, face_name, (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 0, 0), 2)
    return image


def draw_bounding_boxes(image, obj_meta, confidence):
    confidence = '{0:.2f}'.format(confidence)
    rect_params = obj_meta.rect_params
    top = int(rect_params.top)
    left = int(rect_params.left)
    width = int(rect_params.width)
    height = int(rect_params.height)
    obj_name = pgie_classes_str[obj_meta.class_id]
    image = cv2.rectangle(image, (left, top), (left + width, top + height), (0, 0, 255, 0), 2)
    # Note that on some systems cv2.putText erroneously draws horizontal lines across the image
    image = cv2.putText(image, obj_name + ', C=' + str(confidence), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255, 0), 2)
    if obj_meta.class_id == PGIE_CLASS_ID_PERSON:
        # this is a person, let's see if we know this person and draw a box around the face
        # TODO: May be we should crop the image to the size of the object to optimize speed, but let's use the complete image for now
        image = face_recog(image, obj_meta)
    elif obj_meta.class_id == PGIE_CLASS_ID_VEHICLE:
        # this is a vehicle, let's see if we can find a license plate
        log.info('-- Vehicle detected, but no function yet to check his license plate')
    else:
        # some other object, do nothing
        log.info('-- Object detected, but it is not a person or a vehicle')
    return image


def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps = decoder_src_pad.get_current_caps()
    gststruct = caps.get_structure(0)
    gstname = gststruct.get_name()
    source_bin = data
    features = caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not
    # audio
    if(gstname.find("video") != -1):
        # Link the decodebin pad only if decodebin has picked nvidia
        # decoder plugin nvdec_*. We do this by checking if the pad caps contain
        # NVMM memory features.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad = source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick nvidia decoder plugin.\n")


def decodebin_child_added(child_proxy, Object, name, user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added", decodebin_child_added, user_data)
    if(is_aarch64() and name.find("nvv4l2decoder") != -1):
        print("Seting bufapi_version\n")
        Object.set_property("bufapi-version", True)


def create_source_bin(index, uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name = "source-bin-%02d" % index
    print(bin_name)
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin = Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri", uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added", cb_newpad, nbin)
    uri_decode_bin.connect("child-added", decodebin_child_added, nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin, uri_decode_bin)
    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin


def main(args):
    # Check input arguments
    if len(args) < 2:
        sys.stderr.write("usage: %s <uri1> [uri2] ... [uriN] <folder to save frames>\n" % args[0])
        sys.exit(1)

    for i in range(0, len(args) - 2):
        fps_streams["stream{0}".format(i)] = GETFPS(i)
    number_sources = len(args) - 2

    folder_name = args[-1]
    if path.exists(folder_name):
        sys.stderr.write("The output folder %s already exists. Please remove it first.\n" % folder_name)
        sys.exit(1)

    # start logging and counter and create directory for any unknow faces just in case we find any
    logpath = Path(logdir)
    logpath.mkdir(parents=True, exist_ok=True)
    log = init_log(logfile, process, loglevel, logsize, logbackups)
    log.critical('Starting program: %s with OpenCv version %s in %s mode and saving unknow face: %s' % (process, cv2.__version__, 'Screen' if gui else 'Headless', 'On' if save_unknown else 'Off'))
    starttime = time.perf_counter()
    unknow_faces_path = Path(unknow_face_dir)
    unknow_faces_path.mkdir(parents=True, exist_ok=True)

    # opening learned faces file
    log.info(f'- Opening learned faces file: {learnedfile}')
    with open(learnedfile, 'rb') as trainedfacesfile:
        # reading the learned faces file
        Names = pickle.load(trainedfacesfile)
        Sequence = pickle.load(trainedfacesfile)
        Encodings = pickle.load(trainedfacesfile)

    os.mkdir(folder_name)
    log.warning(f'- Frames will be saved in: {folder_name}')
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements */
    # Create Pipeline element that will form a connection of other elements
    log.warning("- Creating Pipeline\n")
    pipeline = Gst.Pipeline()
    is_live = False

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")
    print("Creating streamux \n ")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    pipeline.add(streammux)
    for i in range(number_sources):
        os.mkdir(folder_name + "/stream_" + str(i))
        frame_count["stream_" + str(i)] = 0
        saved_count["stream_" + str(i)] = 0
        print("Creating source_bin ", i, " \n ")
        uri_name = args[i + 1]
        if uri_name.find("rtsp://") == 0:
            is_live = True
        source_bin = create_source_bin(i, uri_name)
        if not source_bin:
            sys.stderr.write("Unable to create source bin \n")
        pipeline.add(source_bin)
        padname = "sink_%u" % i
        sinkpad = streammux.get_request_pad(padname)
        if not sinkpad:
            sys.stderr.write("Unable to create sink pad bin \n")
        srcpad = source_bin.get_static_pad("src")
        if not srcpad:
            sys.stderr.write("Unable to create src pad bin \n")
        srcpad.link(sinkpad)
    print("Creating Pgie \n ")
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")
    # Add nvvidconv1 and filter1 to convert the frames to RGBA
    # which is easier to work with in Python.
    print("Creating nvvidconv1 \n ")
    nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "convertor1")
    if not nvvidconv1:
        sys.stderr.write(" Unable to create nvvidconv1 \n")
    print("Creating filter1 \n ")
    caps1 = Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA")
    filter1 = Gst.ElementFactory.make("capsfilter", "filter1")
    if not filter1:
        sys.stderr.write(" Unable to get the caps filter1 \n")
    filter1.set_property("caps", caps1)
    print("Creating tiler \n ")
    tiler = Gst.ElementFactory.make("nvmultistreamtiler", "nvtiler")
    if not tiler:
        sys.stderr.write(" Unable to create tiler \n")
    print("Creating nvvidconv \n ")
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")
    print("Creating nvosd \n ")
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
    if(is_aarch64()):
        print("Creating transform \n ")
        transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
        if not transform:
            sys.stderr.write(" Unable to create transform \n")

    print("Creating EGLSink \n")
    sink = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")
    if not sink:
        sys.stderr.write(" Unable to create egl sink \n")

    if is_live:
        print("Atleast one of the sources is live")
        streammux.set_property('live-source', 1)

    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', number_sources)
    streammux.set_property('batched-push-timeout', 4000000)
    pgie.set_property('config-file-path', "dstest_imagedata_config.txt")
    pgie_batch_size = pgie.get_property("batch-size")
    if(pgie_batch_size != number_sources):
        print("WARNING: Overriding infer-config batch-size", pgie_batch_size, " with number of sources ", number_sources, " \n")
        pgie.set_property("batch-size", number_sources)
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil((1.0 * number_sources) / tiler_rows))
    tiler.set_property("rows", tiler_rows)
    tiler.set_property("columns", tiler_columns)
    tiler.set_property("width", TILED_OUTPUT_WIDTH)
    tiler.set_property("height", TILED_OUTPUT_HEIGHT)

    sink.set_property("sync", 0)

    if not is_aarch64():
        # Use CUDA unified memory in the pipeline so frames
        # can be easily accessed on CPU in Python.
        mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
        streammux.set_property("nvbuf-memory-type", mem_type)
        nvvidconv.set_property("nvbuf-memory-type", mem_type)
        nvvidconv1.set_property("nvbuf-memory-type", mem_type)
        tiler.set_property("nvbuf-memory-type", mem_type)

    print("Adding elements to Pipeline \n")
    pipeline.add(pgie)
    pipeline.add(tiler)
    pipeline.add(nvvidconv)
    pipeline.add(filter1)
    pipeline.add(nvvidconv1)
    pipeline.add(nvosd)
    if is_aarch64():
        pipeline.add(transform)
    pipeline.add(sink)

    print("Linking elements in the Pipeline \n")
    streammux.link(pgie)
    pgie.link(nvvidconv1)
    nvvidconv1.link(filter1)
    filter1.link(tiler)
    tiler.link(nvvidconv)
    nvvidconv.link(nvosd)
    if is_aarch64():
        nvosd.link(transform)
        transform.link(sink)
    else:
        nvosd.link(sink)

    # create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    tiler_sink_pad = tiler.get_static_pad("sink")
    if not tiler_sink_pad:
        sys.stderr.write(" Unable to get src pad \n")
    else:
        tiler_sink_pad.add_probe(Gst.PadProbeType.BUFFER, tiler_sink_pad_buffer_probe, 0)

    # List the sources
    print("Now playing...")
    for i, source in enumerate(args[:-1]):
        if (i != 0):
            print(i, ": ", source)

    print("Starting pipeline \n")
    # start play back and listed to events
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    # cleanup
    print("Exiting app\n")
    pipeline.set_state(Gst.State.NULL)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
