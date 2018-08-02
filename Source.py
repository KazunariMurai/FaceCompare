# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 17:15:34 2018

@author: User
"""

# https://gist.github.com/alexcasalboni/0f21a1889f09760f8981b643326730ff

import boto3

# Let's use Amazon S3
#s3 = boto3.resource('s3')

# Print out bucket names
#for bucket in s3.buckets.all():
#    print(bucket.name)

# detect faces
BUCKET = "smart119-face-recognition" 
KEY_S = "Aaron_Eckhart_0004.jpg"
KEY_T = "Aaron_Guiel_0001.jpg"
FEATURES_BLACKLIST = ("Landmarks", "Emotions", "Pose", "Quality", "BoundingBox", "Confidence", "AgeRange")

def detect_faces(bucket, key, attributes=['ALL'], region="us-east-2"):
	rekognition = boto3.client("rekognition", region)
	response = rekognition.detect_faces(
	    Image={
			"S3Object": {
				"Bucket": bucket,
				"Name": key,
			}
		},
	    Attributes=attributes,
	)
	return response['FaceDetails']

# Source information
print ("Source face : ")
for face in detect_faces(BUCKET, KEY_S):
	print ("Face ({Confidence}%)".format(**face))
	# emotions
	for emotion in face['Emotions']:
		print ("  {Type} : {Confidence}%".format(**emotion))
	# quality
	for quality, value in face['Quality'].items():
		print ("  {quality} : {value}".format(quality=quality, value=value))
	# facial features
	for feature, data in face.items():
		if feature not in FEATURES_BLACKLIST:
			#print(f())
			print ("  {feature}({data[Value]}) : {data[Confidence]}%".format(feature=feature, data=data))

# Target information
print ("Target face : ")
for face in detect_faces(BUCKET, KEY_T):
	print ("Face ({Confidence}%)".format(**face))
	# emotions
	for emotion in face['Emotions']:
		print ("  {Type} : {Confidence}%".format(**emotion))
	# quality
	for quality, value in face['Quality'].items():
		print ("  {quality} : {value}".format(quality=quality, value=value))
	# facial features
	for feature, data in face.items():
		if feature not in FEATURES_BLACKLIST:
			#print(f())
			print ("  {feature}({data[Value]}) : {data[Confidence]}%".format(feature=feature, data=data))
   
# compare
def compare_faces(bucket, key, bucket_target, key_target, threshold=0, region="us-east-2"):
	rekognition = boto3.client("rekognition", region)
	response = rekognition.compare_faces(
	    SourceImage={
			"S3Object": {
				"Bucket": bucket,
				"Name": key,
			}
		},
		TargetImage={
			"S3Object": {
				"Bucket": bucket_target,
				"Name": key_target,
			}
		},
	    SimilarityThreshold=threshold,
	)
	return response['SourceImageFace'], response['FaceMatches']


source_face, matches = compare_faces(BUCKET, KEY_S, BUCKET, KEY_T)

# the main source face
print ("Source Face ({Confidence}%)".format(**source_face))

# one match for each target face
for match in matches:
	#print ("Target Face ({Confidence}%)".format(**match['Face']))
	print ("  Similarity : {}%".format(match['Similarity']))
