# Intro2CV_final_project
This project is created to develop a robust human detection algorithm for autonomous mobile robot ULTRABOT, using RGB-D images from Intel Realsense D435 camera. 

ULTRABOT is the robot developed to perform disinfection by UVC lamps fully autonomously. However, if a human suddenly appears near the robot UVC lamps, it MUST switch off them in order not to harm the human's eyes. In order to achieve that, a low-level human detection algorithm for sych emergency cases is needed.

Depth channel allows to calculate a distance to a human. If a human stays in front of the lamps, but farther than 10m (outside the dangerous zone) - the robot should not switch the lamps.

### Team
- Stepan Perminov (stepan.perminov@skoltech.ru)
- Alexander Sedunin (alexander.sedunin@skoltech.ru)

<p align="center">
<img src="https://old.sk.ru/resized-image.ashx/__size/550x0/__key/telligent-evolution-components-attachments/13-50-00-00-00-02-16-56/skoltech-rastr-ENG.png" height="40">
</p>
