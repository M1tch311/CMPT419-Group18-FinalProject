import pygame
pygame.init()

# Set up the game window
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("UFO Rush")

# Load the background image
background_image = pygame.image.load("night_sky.jpg")  # Replace with your image filename
# Sourced from: https://pixabay.com/photos/night-moon-mountains-alps-4702174/
# Resize the image to fit the window
background_image = pygame.transform.scale(background_image, (800, 600))

# Load the character image
player_image = pygame.image.load("blue_ufo.png")  # Replace with your image file name
# Sourced from: https://opengameart.org/content/basic-ufo-set)

# Scale the image
player_image = pygame.transform.scale(player_image, (50, 50))  # Resize to 50x50 pixels

# Player settings
player_x = 10
player_y = 550
player_speed = 0.2

# Enemy settings
enemy_1_x = 750
enemy_1_y = 550
enemy_speed = 0.2

# Scoring system
score = 0
# lives = 3

# Mode corresponding to expression, 1 is easy, 2 is medium, 3 is hard
mode = 1

# Collision detection structure
def is_collision(player_x, player_y, enemy_x, enemy_y):
    distance = ((player_x - enemy_x)**2 + (player_y - enemy_y)**2) ** 0.5
    return distance < 50


# Game loop
running = True
while running:

    # Check for player input
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check for collision

   # if is_collision(player_x, player_y, enemy_x, enemy_y):
       # print("Game Over!")
      #  running = False

    if is_collision(player_x, player_y, enemy_1_x, enemy_1_y):
        #player_x -= player_speed
        enemy_1_x += enemy_speed
        player_y -= player_speed


        #lives -= 1
        #if lives == 0:
            #print("Game Over!")
            #running = False
        #else:
            # Reset positions
            #player_x = 370
            #enemy_x = 200

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
        player_y -= 2*player_speed
    # if keys[pygame.K_DOWN]:
        # player_y += player_speed

    # Border constraints
    if player_x < 0:
        player_x += player_speed
    if player_x > 750 or (is_collision(player_x, player_y, enemy_1_x, enemy_1_y) == True):
        player_x -= player_speed

    if player_y < 550:
        player_y += player_speed
    if player_y < 0:
        player_y += player_speed

    # Fill the screen and draw player and enemy
    # screen.fill((0, 0, 255))
    # Draw the background in the game loop
    screen.blit(background_image, (0, 0))  # Draw at the top-left corner (0, 0)
    screen.blit(player_image, (player_x, player_y))  # Use player_x and player_y for positioning
    # pygame.draw.rect(screen, (255, 0, 0), (player_x, player_y, 50, 50))
    pygame.draw.rect(screen, (0, 255, 0), (enemy_1_x, enemy_1_y, 50, 50))

    # Moving the enemy
    enemy_1_x -= enemy_speed
    # if enemy_x >= 750 or enemy_x <= 0:
    if enemy_1_x <= -50:
        enemy_1_x = 800  # Reset position to give illusion of new obstacle
        if mode == 1:
            enemy_speed = 0.2
        elif mode == 2:
            enemy_speed = 0.4
        elif mode == 3:
            enemy_speed = 0.5
        score += 800
        print(f"Score: {score}")

    pygame.display.update()

pygame.quit()

#To-do:
#Implement code to detect facial expression and change obstacle speed and appearance accordingly
#Implement timer as end condition
#Improve obstacles
#If possible, use assets to polish game appearance