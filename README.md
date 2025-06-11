# movie-recommender-system

This is a Recommendation System designed to automatically generate recommendations for films. 

The project is completed during the preparation of Elizaveta A. Suchanova's bachelor thesis at SPbPU Institute of Computer Science and Cybersecurity (SPbPU ICSC).


Authors and Contributors
Advisor and minor contributor: Vladimir A. Parkhomenko Senior Lecturer at SPbPU ICSC

Main Contributor: Elizaveta A. Suchanova Student at SPbPU ICSC

Reference (to be updated after publication):
Please, using this repository, cite the following paper https://hsse.spbstu.ru/userfiles/files/1941_sovremennie_tehnologii_s_oblozhkoy.pdf, 2025

## SETUP

To run the system, a training dataset containing ratings, films and and their metadata must be downloaded via a link: https://drive.google.com/file/d/1f4N-4_zvRABeLBjyMdnDGF_giKRq4O_I/view?usp=drive_link
The data is presented as a MySQL database dump.

First you need to download all the required packages from the requirements.txt file.
All variables related to paths for saving data, database configuration, API and user registration tokens are contained in the config.py file.
The system and server can be started via the command `python -m api.main`.
At the first run you will have to wait for a considerable time, as the models are trained for about an hour and a half, at subsequent runs they will already be loaded from the file.
