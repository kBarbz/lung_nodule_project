zip -FF subset1.zip -O subset1.fixed.zip
unzip subset1.fixed.zip -d datasets/luna16/images/

zip -FF subset2.zip -O subset2.fixed.zip
unzip subset2.fixed.zip -d datasets/luna16/images/

zip -FF subset3.zip -O subset3.fixed.zip
unzip subset3.fixed.zip -d datasets/luna16/images/

zip -FF subset4.zip -O subset4.fixed.zip
unzip subset4.fixed.zip -d datasets/luna16/images/

zip -FF subset5.zip -O subset5.fixed.zip
unzip subset5.fixed.zip -d datasets/luna16/images/

zip -FF subset6.zip -O subset6.fixed.zip
unzip subset6.fixed.zip -d datasets/luna16/images/

zip -FF subset7.zip -O subset7.fixed.zip
unzip subset7.fixed.zip -d datasets/luna16/images/

zip -FF subset8.zip -O subset8.fixed.zip
unzip subset8.fixed.zip -d datasets/luna16/images/

zip -FF subset9.zip -O subset9.fixed.zip
unzip subset9.fixed.zip -d datasets/luna16/images/

mv ./annotations.csv datasets/luna16/
mv ./candidates.csv datasets/luna16/
mv ./candidates_V2.zip datasets/luna16/
mv ./sampleSubmission.csv datasets/luna16/
mv ./evaluationScript.zip datasets/luna16/
mv ./seg-lungs-LUNA16.zip datasets/luna16/

mv ./efficientdet-d0.pth ./efficientdet_repo/weights
mv ./efficientdet-d1.pth ./efficientdet_repo/weights
mv ./efficientdet-d2.pth ./efficientdet_repo/weights
mv ./efficientdet-d3.pth ./efficientdet_repo/weights
mv ./efficientdet-d4.pth ./efficientdet_repo/weights
mv ./efficientdet-d5.pth ./efficientdet_repo/weights
mv ./efficientdet-d6.pth ./efficientdet_repo/weights
mv ./efficientdet-d7.pth ./efficientdet_repo/weights

#Expand-Archive subset0.zip -d datasets/luna16/images/
#Expand-Archive subset1.zip -d datasets/luna16/images/
#Expand-Archive subset2.zip -d datasets/luna16/images/
#Expand-Archive subset3.zip -d datasets/luna16/images/
#Expand-Archive subset4.zip -d datasets/luna16/images/
#Expand-Archive subset5.zip -d datasets/luna16/images/
#Expand-Archive subset6.zip -d datasets/luna16/images/
#Expand-Archive subset7.zip -d datasets/luna16/images/
#Expand-Archive subset8.zip -d datasets/luna16/images/
#Expand-Archive subset9.zip -d datasets/luna16/images/
#Move-Item ./annotations.csv datasets/luna16/images/
#Move-Item ./candidates.csv datasets/luna16/images/
#Move-Item ./candidates_V2.zip datasets/luna16/images/
#Move-Item ./evaluationScript.zip datasets/luna16/images/
#Move-Item ./sampleSubmission.csv datasets/luna16/images/
#Move-Item ./seg-lungs-LUNA16.zip datasets/luna16/images/
