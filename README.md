# 🚭 Smoking Detection with Computer Vision


This beginner-friendly project uses basic image processing and motion analysis techniques to identify smoking gestures in a video. The goal is to provide a lightweight, dataset-free approach that can later be enhanced with machine learning or deep learning models.

## 📽️ Demo

▶️ [Watch Demo Video](https://drive.google.com/file/d/1WmtxETFAkYsbXaRbmHAl8xohkaCa48wy/preview)

This test video demonstrates how the system detects smoking-like behavior in a pre-recorded clip.

## 🧠 How It Works

- 📽️ Reads the video **frame-by-frame** using OpenCV.
- 🔄 Applies basic **frame differencing** and/or **color detection** logic.
- 🚬 Flags **smoking-like motion patterns or zones** visually.
- 🎯 Highlights the area on screen if **smoking behavior** is suspected.

> ⚠️ This is a **basic prototype** and may produce **false positives**.  
> It can be improved by training a **deep learning model** for more accurate results.
