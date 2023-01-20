import pygame
import sys
import os
import cv2
import numpy as np

import tensorflow as tf

BACKGROUND = (0,0,0)
WIDTH = 800
HEIGHT = 560

FPS = 360

path = os.path.dirname(os.path.realpath(__file__))
cpPath = path + "\mnistcallback"

fpsClock = pygame.time.Clock()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont('freesansbold.ttf', 35)

def main():

    # background of panel
    screen.fill(BACKGROUND)

    dragging = False

    # load the number recognization model
    model = tf.keras.models.load_model(cpPath)
    model.summary()

    score = {}
    for i in range(0, 10):
        score[i] = 0

    currMax = None
    while True :

        # side bar
        pygame.draw.rect(screen, (36, 36, 36), pygame.Rect(560, 0, 240, 560))

        # Get inputs
        for event in pygame.event.get() :
            if event.type == pygame.QUIT :
                
                # remove the image used for processing 
                os.remove(path + "\current.jpeg")
                # quit the program
                pygame.quit()
                sys.exit()
            
            # mouse button down, dragging can occur
            elif event.type == pygame.MOUSEBUTTONDOWN:
                
                # left click
                if event.button == 1:
                    if pygame.mouse.get_pos()[0] <= 560 and pygame.mouse.get_pos()[0] <= 560:
                        dragging = True

                # right click ( reset the canvas )
                elif event.button == 3:

                    dragging = False
                    # reset the screen
                    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, 0, 560, 560))
                    # remove the predicted value
                    currMax = None
                    # if nothing on screen, all score values are 0
                    for i in range(0, 10):
                        score[i] = 0

            # mouse button up, dragging cannot occur
            elif event.type == pygame.MOUSEBUTTONUP:
                
                if dragging:
                    # get the current screen of pygame for processing
                    # process the image and scale it to 28 x 28
                    ImgProcessing(screen)
                    # get the current image
                    img = cv2.imread("current.jpeg", 0)
                    # get bit value of image
                    img_reverted= cv2.bitwise_not(img)

                    # invert the bit values
                    inputimg = []
                    for i in img_reverted:
                        temp = []
                        for j in i:
                            temp.append(float((255-j)/255))
                        inputimg.append(temp)

                    # reshape the input
                    inputimg = [np.array(inputimg).reshape(28,28,1)]
                    # predict number on the current screen
                    predict = model.predict(np.array(inputimg))
                    # add up all the scores
                    total = sum(predict[0])

                    # get the percentage for each number
                    for x, i in enumerate(predict[0]):
                        score[x] = round((i / total) * 100, 3)

                    # predicted value
                    currMax = np.argmax(predict[0])

                dragging = False

            # mouse is moving and mouse button is down, draw on the screen
            elif event.type == pygame.MOUSEMOTION:

                if dragging:
                    # get position and draw 
                    pos = pygame.mouse.get_pos()

                    if pos[0] <= 560 and pos[1] <= 560:
                        # pygame.draw.rect(screen, (255,255,255), pygame.Rect(pos[0] - 25, pos[1] - 25, 50, 50))
                        pygame.draw.circle(screen, (255,255,255),(pos[0], pos[1]), 20)

            # interface
            for i in score:
                # color of font
                if i == currMax:
                    color = (255, 215, 0)
                else:
                    color = (255,255,255)

                # number : probability
                text = font.render(f'{i} : {float(score[i])}%', False, color)
                screen.blit(text, (580, 15 + (i * 40)))

                if i == 9:
                    # predicted value
                    text = font.render(f'Predicted value:', False, (255,255,255))
                    screen.blit(text, (580, 15 + (11 * 40)))

                    text = font.render(f'{currMax}', False, (255,255,255))
                    screen.blit(text, (580, 15 + (12 * 40)))
                     

            pygame.display.update()
            fpsClock.tick(FPS)

    
def ImgProcessing(screen):
    """
    Process the image so it is suitable to be inputed into neural network
    """
    # save the image
    pygame.image.save(screen, "current.jpeg")
    # read the image
    img = cv2.imread("current.jpeg")
    # crop the screen to 560 x 560
    croppedImg = img[0:560, 0:560]
    # blur the image
    blurredImg = cv2.blur(croppedImg,(10,10))
    # rescale image to 28 x 28
    rescaledImg = cv2.resize(blurredImg, (28, 28), 0.05, 0.05)
    # save the processed image
    cv2.imwrite("current.jpeg", rescaledImg)


main()