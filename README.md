# CE-4302_Taller4

## SAXPY program
Para compilar el programa que aplica la operación SAXPY tanto serial (normal) como paralelo (OpenMP + NEON), dentro del directorio ./SAXPY/jni/ se ejecuta:
```console
~$ adb push ../libs/armeabi-v7a/saxpy /data/local/tmp
~$ adb shell /data/local/tmp/saxpy
```


## Dot Product
Para compilar la aplicación que aplica la operación el producto punto tanto serial (normal) como paralelo (OpenMP + NEON), dentro del directorio ./DotProduct/jni/ se ejecuta:
```console
~$ adb push ../libs/armeabi-v7a/dot_product /data/local/tmp
~$ adb shell /data/local/tmp/dot_product
```

