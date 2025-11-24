# ---------- COCOStuff ----------
# Follow
# https://github.com/nightrome/cocostuff
# Download everything
mkdir COCOStuff
cd COCOStuff
wget --directory-prefix=downloads http://images.cocodataset.org/zips/train2017.zip
wget --directory-prefix=downloads http://images.cocodataset.org/zips/val2017.zip
wget --directory-prefix=downloads http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip

# Unpack everything
mkdir -p dataset/images
mkdir -p dataset/annotations
unzip downloads/train2017.zip -d dataset/images/
unzip downloads/val2017.zip -d dataset/images/
unzip downloads/stuffthingmaps_trainval2017.zip -d dataset/annotations/

# https://github.com/xu-ji/IIC/blob/master/datasets/README.txt
wget https://www.robots.ox.ac.uk/~xuji/datasets/COCOStuff164kCurated.tar.gz
tar -xzf COCOStuff164kCurated.tar.gz
mv COCO/COCOStuff164k ./currated
rmdir COCO
rm COCOStuff164kCurated.tar.gz

# ---------- Cityscapes ----------
# https://github.com/cemsaz/city-scapes-script
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=XXXX&password=XXX&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
unzip gtFine_trainvaltest.zip 
unzip leftImg8bit_trainvaltest.zip


# ---------- VOC ----------
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
# generate folder VOCdevkit/VOC2012
tar -xvf VOCtrainval_11-May-2012.tar

# ---------- ADE20K ----------
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip


# ---------- DAVIS ----------
mkdir DAVIS
cd DAVIS
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip
rm DAVIS-2017-trainval-480p.zip
mv DAVIS/* ./
rmdir DAVIS
