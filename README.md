# Tibia HPnMP Detector




Image processing program to read HP and MP from Tibia game using only Image Processing

Steps:
1 - Take a screenshot from the game
2 - Using a Haar Cascade to indentify HP and MP bars (trained with opencv Train Cascade)
3 - Use raw Image processing to segment the numbers
4 - Use a Sklearn classifier to recognize each segmented digit
5 - Once recognized, display the results


