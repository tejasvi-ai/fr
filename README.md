## Setting up LFW dataset


```python
!wget http://vis-www.cs.umass.edu/lfw/lfw.tgz -q
```


```python
!tar -xf lfw.tgz
```

#### Optionally remove exceedingly frequent examples


```python
# !cd lfw && du --inodes -S | sort -rh | sed -n '1,50{/^.\{71\}/s/^\(.\{30\}\).*\(.\{37\}\)$/\1...\2/;p}' > f
# !cd lfw && sed -i 's/.*\.\/\(.*\)/\1/' f
# !mkdir tmp
# !cd lfw && xargs mv -t ../tmp < f
# !rm lfw/f
```

#### Create zip archive for LFW dataset


```python
!cd lfw && zip -r -q ../lfw.zip *
```

## SphereFace


```python
!git clone --depth=1 https://github.com/clcarwin/sphereface_pytorch sphereface
```

    Cloning into 'sphereface'...
    remote: Enumerating objects: 18, done.[K
    remote: Counting objects: 100% (18/18), done.[K
    remote: Compressing objects: 100% (16/16), done.[K
    remote: Total 18 (delta 0), reused 15 (delta 0), pack-reused 0[K
    Unpacking objects: 100% (18/18), done.



```python
!sudo apt install atool -y -qq
!cd sphereface/model && atool -x -q *.7z
```


```python
!python3 lfw_eval.py --model sphereface/model/sphere20a_20171020.pth --lfw 'lfw.zip'
```

## Deepface package


```python
!pip3 install deepface -qq
```

#### [`deepface`](https://github.com/serengil/deepface/) package provides VGG-Face , Google FaceNet, OpenFace, Facebook DeepFace and DeepID<br>
The helper script using `deepface` package is [DeepFace.py](DeepFace.py).


```python
!python3 lfw_eval.py --net $MODEL_NAME --lfw 'lfw.zip'
```

Replace $MODEL_NAME with one of `deepface`, `openface`, and `facenet`.

