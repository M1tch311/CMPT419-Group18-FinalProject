import pygame
import sys
import queue
from threading import Thread
import random
import main

pygame.init()



# Set up the game window
# window_x = 800
# window_y = 600
window_x = 1000
window_y = 800
screen = pygame.display.set_mode((window_x, window_y))
pygame.display.set_caption("UFO Rush")

# Load the background image
background_image = pygame.image.load("game_assets/night_sky.jpg")
# Sourced from jplenio on pixabay: https://pixabay.com/photos/night-moon-mountains-alps-4702174/
# Resize the image to fit the window
background_image = pygame.transform.scale(background_image, (window_x, window_y))

# Load the character image
player_image = pygame.image.load("game_assets/blue_ufo.png")
# Illustration by The_Scientist___ from: https://opengameart.org/content/basic-ufo-set)

# Scale the image
player_image = pygame.transform.scale(player_image, (50, 50))  # Resize to 50x50 pixels

pillar_image = pygame.image.load("game_assets/pillar.png")
inverted_pillar_image = pygame.image.load("game_assets/inverted_pillar.png")
# Pillar image illustration by SilviaP_Design on pixabay: https://pixabay.com/illustrations/pillar-stone-old-ruin-building-1183349/

# Scale the images
pillar_image = pygame.transform.scale(pillar_image, (50, int(window_y * 0.75)))
inverted_pillar_image = pygame.transform.scale(inverted_pillar_image, (50, int(window_y * 0.75)))

# Player settings (position)
player_x = 10
player_y = int(window_y * 0.5)
player_speed = 3  # 0.2

gap_measure = int(window_y * 0.125)

# obstacle settings (positions)
obstacle_1_x = window_x
obstacle_1_y = int(window_y * 0.5) + gap_measure
obstacle_1_height = int(window_y * 0.75)
ceiling_1 = 0
# obstacle_speed = 2  # 0.2
# ceiling for obstacle 1 is the value 0 (ground obstacle)

obstacle_2_height = int(window_y * 0.75)
obstacle_2_x = window_x
obstacle_2_y = (obstacle_2_height - int(window_y * 0.5)) * -1 - gap_measure
ceiling_2 = 1
# ceiling for obstacle 2 is the value 1 (ceiling obstacle)

# second obstacle attached to the ceiling
obstacle_3_x = int(1.5 * window_x)
obstacle_3_y = 0
obstacle_3_height = int(window_y * 0.75)
ceiling_3 = 1

# second obstacle from the ground
obstacle_4_x = window_x
obstacle_4_y = window_y - int(window_y * 0.75)
obstacle_4_height = int(window_y * 0.75)
ceiling_4 = 0

# Scoring system and timer
score = 0
game_timer = 5 * 60 * 1000  # game duration is five minutes (in milliseconds)
start_time = pygame.time.get_ticks()
# spawn_interval = 1500  # interval for obstacles to spawn
# next_spawn_time = pygame.time.get_ticks() + spawn_interval
detect_interval = 10 * 1000
next_detect_time = pygame.time.get_ticks() + detect_interval

game_over = False

# Pause feature settings
paused = False
pause_button = pygame.Rect(window_x - 100, 10, 80, 40)

# Font to display score and other text
score_font = pygame.font.SysFont(None, 36)
game_over_font = pygame.font.SysFont(None, 72)

# Mode corresponding to expression, 1 is easy, 2 is medium, 3 is hard
mode = 1

def draw_buttons():
    restart_button = pygame.Rect(int(window_x * 0.5 - 100), int(window_y * 0.5 + 50), 200, 50)
    quit_button = pygame.Rect(int(window_x * 0.5 - 100), int(window_y * 0.5 + 120), 200, 50)

    pygame.draw.rect(screen, (0, 255, 0), restart_button)
    restart_text = score_font.render("Restart", True, (0, 0, 0))
    restart_rect = restart_text.get_rect(center=restart_button.center)
    screen.blit(restart_text, restart_rect)

    pygame.draw.rect(screen, (255, 0, 0), quit_button)
    quit_text = score_font.render("Quit", True, (0, 0, 0))
    quit_rect = quit_text.get_rect(center=quit_button.center)
    screen.blit(quit_text, quit_rect)

    return restart_button, quit_button

# Collision detection structure
def is_collision(player_x_coord, player_y_coord, obstacle_x, obstacle_y, obstacle_height, ceiling):
    collide = False
    distance_x = (((player_x_coord + 25) - (obstacle_x + 25)) ** 2) ** 0.5
    if ceiling == 0 and distance_x < 50:
        distance_y = (((player_y_coord + 25) - (obstacle_y + 25)) ** 2) ** 0.5
        if (player_y_coord > obstacle_y) or distance_y < 50:
            collide = True

    if ceiling == 1 and distance_x < 50:
        # body = obstacle_y + obstacle_2_height
        # distance_y = ((player_y_coord - body) ** 2) ** 0.5
        # if obstacle_y < player_y_coord < (obstacle_y + obstacle_height):  # or distance_y <= 0
        if 0 < player_y_coord < (obstacle_y + obstacle_height):
            collide = True

    return collide

# def is_collision(player_x, player_y, obstacle_x, obstacle_y):
#     distance = ((player_x - obstacle_x)**2 + (player_y - obstacle_y)**2) ** 0.5
#     return distance < 50

# IMPORTANT!!!!
# Code to detect facial expressions and select mode accordingly needs to be added
def game_loop(emotion):

    que = queue.LifoQueue()

    clock = pygame.time.Clock()

    flag = 0

    global obstacle_1_x, obstacle_1_y, obstacle_1_height, player_x, player_y, score, game_timer, player_speed, game_over, paused
    global obstacle_2_x, obstacle_2_y, obstacle_2_height, obstacle_3_x, obstacle_3_y, obstacle_3_height
    global obstacle_4_x, obstacle_4_y, obstacle_4_height, start_time, next_detect_time, detect_interval, mode

    t1 = Thread(target=main.runcameraclassification, args=(que,))
    t1.daemon = True
    t1.start()

    # Game loop
    while True:

        # if flag == 0:
        #     t1.start()
        #     flag = 1

        clock.tick(60)

        # Check for player input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                # t1.join()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if pause_button.collidepoint(mouse_pos):
                    paused = not paused

                if paused or game_over:
                    restart_button, quit_button = draw_buttons()  # Get the buttons
                    if restart_button.collidepoint(mouse_pos):
                        # Restart the game by resetting key variables
                        player_x = 10
                        player_y = int(window_y * 0.5)
                        score = 0
                        # obstacles = []
                        obstacle_1_x = window_x
                        obstacle_2_x = window_x
                        obstacle_3_x = window_x
                        obstacle_4_x = window_x
                        start_time = pygame.time.get_ticks()
                        # next_spawn_time = pygame.time.get_ticks() + spawn_interval
                        next_detect_time = pygame.time.get_ticks() + detect_interval
                        game_over = False
                        paused = False
                    elif quit_button.collidepoint(mouse_pos):
                        # t1.join()
                        pygame.quit()
                        sys.exit()

        # Check for collision
        # if is_collision(player_x, player_y, obstacle_x, obstacle_y):
            # print("Game Over!")
            # running = False

        if not game_over and not paused:
            if is_collision(player_x, player_y, obstacle_1_x, obstacle_1_y, obstacle_1_height, ceiling_1):
                obstacle_1_x += obstacle_speed
                if obstacle_2_x == obstacle_1_x:
                    obstacle_2_x += obstacle_speed
                player_y -= player_speed

            if is_collision(player_x, player_y, obstacle_2_x, obstacle_2_y, obstacle_2_height, ceiling_2):
                obstacle_2_x += obstacle_speed
                if obstacle_1_x == obstacle_2_x:
                    obstacle_1_x += obstacle_speed
                # player_y -= player_speed
                player_x -= 2 * player_speed

            if is_collision(player_x, player_y, obstacle_3_x, obstacle_3_y, obstacle_3_height, ceiling_3):
                obstacle_3_x += obstacle_speed
                # player_y -= player_speed
                player_x -= 2 * player_speed

            if is_collision(player_x, player_y, obstacle_4_x, obstacle_4_y, obstacle_4_height, ceiling_4):
                obstacle_4_x += obstacle_speed
                player_y -= player_speed

                # player_x -= player_speed
                # lives -= 1
                # if lives == 0:
                # print("Game Over!")
                # running = False
                # else:
                # Reset positions
                # player_x = 370
                # obstacle_x = 200

            # Increment score
            # score += 1
            # print(f"Score: {score} Lives: {lives}")

            # Moving the player
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                player_x -= player_speed
            if keys[pygame.K_RIGHT]:
                player_x += player_speed
            if keys[pygame.K_SPACE]:
                player_y -= 2 * player_speed
            # if keys[pygame.K_DOWN]:
            # player_y += player_speed

            # Border constraints
            if player_x < 0:
                player_x += player_speed
            if player_x > window_x - 50:
                player_x -= player_speed
            # if (is_collision(player_x, player_y, obstacle_1_x, obstacle_1_y, obstacle_1_height, 0) == True):
                # player_x -= player_speed
            # if (is_collision(player_x, player_y, obstacle_2_x, obstacle_2_y, obstacle_2_height, 1) == True):
                # player_x += player_speed
            if player_y < window_y - 50:
                player_y += player_speed
            if player_y < 0:
                player_y += player_speed

            current_time = pygame.time.get_ticks()
            if current_time - start_time >= game_timer:
                game_over = True

            # Moving the obstacle
            # if current_time - next_detect_time >= detect_interval:
            #    mode = current_emotion
            #    next_detect_time = current_time

            if current_time >= next_detect_time:
                # print(t1.join())
                emo_check = que.get()
                # flag = 0
                # print("The current emotion is: ", ['angry', 'happy', 'neutral'][emo_check])
                if emo_check == 0 and mode > 1:
                    mode -= 1
                    print("The current emotion is: angry")
                elif emo_check == 2 and mode < 3:
                    mode += 1
                    print("The current emotion is: neutral")
                else:
                    print("The current emotion is: happy")
                next_detect_time = current_time + detect_interval

            if mode == 1:
                obstacle_speed = 3  # 0.2
                obstacle_1_x -= obstacle_speed
                obstacle_2_x -= obstacle_speed
                obstacle_2_x = obstacle_1_x
                obstacle_3_x = window_x
                obstacle_4_x = window_x

                if obstacle_1_x <= -50:
                    obstacle_1_x = window_x  # Reset position to give illusion of new obstacle
                    score += window_x  # 800
                    print(f"Score: {score}")
                if obstacle_2_x <= -50:
                    obstacle_2_x = window_x
                    score += window_x  # 800
                    print(f"Score: {score}")

            elif mode == 2:
                obstacle_speed = 5  # 0.4
                obstacle_1_x -= obstacle_speed
                obstacle_2_x -= obstacle_speed
                obstacle_3_x -= obstacle_speed
                obstacle_2_x = obstacle_1_x
                obstacle_4_x = window_x
                if -50 < obstacle_1_x <= obstacle_3_x:
                    obstacle_3_x = abs(obstacle_1_x) + int(window_x * 0.5) + 25
                elif obstacle_3_x < obstacle_1_x:
                    obstacle_1_x = abs(obstacle_3_x) + int(window_x * 0.5) + 25

                if obstacle_1_x <= -50:
                    obstacle_1_x = window_x  # Reset position to give illusion of new obstacle
                    score += window_x  # 800
                    print(f"Score: {score}")
                if obstacle_2_x <= -50:
                    obstacle_2_x = window_x
                    score += window_x  # 800
                    print(f"Score: {score}")
                if obstacle_3_x <= -50:
                    obstacle_3_x = window_x
                    score += window_x  # 800
                    print(f"Score: {score}")

            elif mode == 3:
                obstacle_speed = 5  # 0.5
                obstacle_4_x -= obstacle_speed
                obstacle_1_x -= obstacle_speed
                obstacle_2_x -= obstacle_speed
                obstacle_3_x -= obstacle_speed
                obstacle_2_x = obstacle_1_x
                if -50 <= obstacle_4_x <= obstacle_1_x:
                    obstacle_1_x = abs(obstacle_4_x) + int(window_x * 0.75)
                elif obstacle_4_x > obstacle_1_x:
                    obstacle_4_x = abs(obstacle_3_x) + int(window_x * 0.5) + 25
                if -50 <= obstacle_1_x <= obstacle_3_x:
                    obstacle_3_x = abs(obstacle_1_x) + int(window_x * 0.75)
                elif obstacle_3_x < obstacle_1_x:
                    obstacle_1_x = abs(obstacle_4_x) + int(window_x * 0.5) + 25
                if -50 < obstacle_3_x <= obstacle_4_x:
                    obstacle_4_x = abs(obstacle_3_x) + int(window_x * 0.75)
                elif obstacle_3_x > obstacle_4_x:
                    obstacle_3_x = abs(obstacle_1_x) + int(window_x * 0.5) + 25

                if obstacle_1_x <= -50:
                    obstacle_1_x = window_x  # Reset position to give illusion of new obstacle
                    score += window_x  # 800
                    print(f"Score: {score}")
                if obstacle_2_x <= -50:
                    obstacle_2_x = window_x
                    score += window_x  # 800
                    print(f"Score: {score}")
                if obstacle_3_x <= -50:
                    obstacle_3_x = window_x
                    score += window_x  # 800
                    print(f"Score: {score}")
                if obstacle_4_x <= -50:
                    obstacle_4_x = window_x
                    score += window_x  # 800
                    print(f"Score: {score}")


        # Drawing section

        # Fill the screen and draw player and obstacles
        # screen.fill((0, 0, 255))
        # Draw the background in the game loop
        screen.blit(background_image, (0, 0))  # Draw in the top-left corner (0, 0)
        screen.blit(player_image, (player_x, player_y))  # Use player_x and player_y for positioning
        # pygame.draw.rect(screen, (255, 0, 0), (player_x, player_y, 50, 50))
        # pygame.draw.rect(screen, (0, 255, 0), (obstacle_1_x, obstacle_1_y, 50, obstacle_1_height))
        # pygame.draw.rect(screen, (255, 0, 0), (obstacle_2_x, obstacle_2_y, 50, obstacle_2_height))
        # pygame.draw.rect(screen, (0, 0, 255), (obstacle_3_x, obstacle_3_y, 50, obstacle_3_height))
        # pygame.draw.rect(screen, (0, 0, 255), (obstacle_4_x, obstacle_4_y, 50, obstacle_4_height))
        screen.blit(pillar_image, (obstacle_1_x, obstacle_1_y))
        screen.blit(inverted_pillar_image, (obstacle_2_x, obstacle_2_y))
        screen.blit(inverted_pillar_image, (obstacle_3_x, obstacle_3_y))
        screen.blit(pillar_image, (obstacle_4_x, obstacle_4_y))

        # Drawing the pause button
        pause_button_color = (100, 100, 100)
        pygame.draw.rect(screen, pause_button_color, pause_button)
        if paused:
            pause_text = "Resume"
        else:
            pause_text = "Pause"
        pause_label = score_font.render(pause_text, True, (255, 255, 255))
        pause_label_rect = pause_label.get_rect(center=pause_button.center)
        screen.blit(pause_label, pause_label_rect)

        if not game_over:
            score_text = score_font.render(f"Score: {score}", True, (255, 255, 255))
            screen.blit(score_text, (10, 10))
            time_left = max(0, (game_timer - (pygame.time.get_ticks() - start_time)) // 1000 )
            timer_text = score_font.render(f"Time Left: {time_left}s", True, (255, 255, 255))
            screen.blit(timer_text, (10, 40))

            if paused:
                pause_overlay = pygame.Surface((window_x, window_y))
                pause_overlay.set_alpha(150)
                pause_overlay.fill((0, 0, 0))
                screen.blit(pause_overlay, (0, 0))
                paused_text = game_over_font.render("Paused", True, (255, 255, 255))
                paused_rect = paused_text.get_rect(center=(int(window_x * 0.5), int(window_y * 0.5 - 100)))
                screen.blit(paused_text, paused_rect)
                draw_buttons()
        else:
            overlay = pygame.Surface((window_x, window_y))
            overlay.set_alpha(150)
            overlay.fill((0, 0, 0))
            screen.blit(overlay, (0, 0))
            game_over_text = game_over_font.render("Game Over", True, (255, 255, 255))
            game_over_rect = game_over_text.get_rect(center=(int(window_x * 0.5), int(window_y * 0.5 - 100)))
            screen.blit(game_over_text, game_over_rect)
            final_score_text = score_font.render(f"Final Score: {score}", True, (255, 255, 255))
            final_score_rect = final_score_text.get_rect(center=(int(window_x * 0.5), int(window_y * 0.5 - 40)))
            screen.blit(final_score_text, final_score_rect)
            # Display Restart and Quit buttons
            draw_buttons()

        pygame.display.update()

    t1.join()


# To-do:
# Implement code to detect facial expression and change obstacle speed and appearance accordingly
# Implement timer as end condition
# Improve obstacles
# If possible, use assets to polish game appearance
