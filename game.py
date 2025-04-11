import pygame
import sys
import random
from utils import Vec2d
import config
import os

# CONSTANTS
# Set up the game window
# window_x = 800
# window_y = 600
window_x = 1000
window_y = 800
pillar_height = int(window_y * 0.75)
pillar_width = 75
gap_measure = 50
player_size = Vec2d(50, 50)
difficulty = 0
top_of_screen = 30

asset_folder = 'assets'

# Credits
# Missing image credit: https://developer.valvesoftware.com/wiki/Missing_content
# Pillar image illustration by SilviaP_Design on pixabay: https://pixabay.com/illustrations/pillar-stone-old-ruin-building-1183349/
# Illustration by The_Scientist___ from: https://opengameart.org/content/basic-ufo-set)
# Sourced from jplenio on pixabay: https://pixabay.com/photos/night-moon-mountains-alps-4702174/

class Entity(pygame.sprite.Sprite):
    def __init__(self, pos: Vec2d=Vec2d(0,0), size: Vec2d=Vec2d(0,0), velocity: Vec2d=Vec2d(0,0), image_path:str="default.png"):
        super(Entity, self).__init__()
        self.pos = pos
        self.velocity = velocity
        self.size = size
        if image_path == "":
            self.image = pygame.Surface(size.getTuple())
        else:
            self.raw_image = pygame.image.load(os.path.join(asset_folder, image_path))
            self.image = pygame.transform.scale(self.raw_image, size.getTuple()).convert_alpha()
        self.rect = self.image.get_rect()
    
    def update(self):
        self.pos += self.velocity
        self.rect = pygame.Rect(self.pos.getTuple(), self.size.getTuple())
    
    def setPos(self, pos=Vec2d(0,0)):
        self.pos = pos
    
    def setVelocity(self, velocity=Vec2d(0,0)):
        self.velocity = velocity
    
    def setVelocity(self, x=0, y=0):
        self.velocity = Vec2d(x,y)
    
    def getBounds(self):
        return [self.pos.x, self.pos.x + self.size.x, self.pos.y, self.pos.y + self.size.y]

    def setRight(self, x):
        self.pos.x = x - self.size.x
    
    def setLeft(self, x):
        self.pos.x = x
    
    def setTop(self, y):
        self.pos.y = y
    
    def setBottom(self, y):
        self.pos.y = y - self.size.y

class Obstacle(Entity):
    def __init__(self, pos = Vec2d(0, 0), size = Vec2d(pillar_width, pillar_height), velocity = Vec2d(0, 0), image_path = "pillar.png"):
        super(Obstacle, self).__init__(pos, size, velocity, image_path)
    def update(self):
        self.pos += Vec2d(int(-4 * (difficulty + 1)), 0)
        self.rect = pygame.Rect(self.pos.getTuple(), self.size.getTuple())

class ScoreObject(Obstacle):
    def __init__(self, pos = Vec2d(0, 0), size = Vec2d(pillar_width//2, gap_measure), velocity = Vec2d(0, 0), image_path = ""):
        super().__init__(pos, size, velocity, image_path)

def createObstacle(center_pos):
    top = Obstacle(
        pos=(center_pos + Vec2d(0, gap_measure))
    )
    bottom = Obstacle(
        pos=(center_pos - Vec2d(0, gap_measure + pillar_height))
    )
    bottom.image = pygame.transform.rotate(bottom.image, 180)
    counter = ScoreObject(
        pos=(center_pos - Vec2d(-pillar_width/4, gap_measure//2))
    )
    return (top, bottom, counter)

def bound_to_screen(entity:Entity):
    if entity.pos.x < 0:
        entity.pos.x = 0
    if entity.pos.x > (window_x - entity.size.x):
        entity.pos.x = window_x - entity.size.x
    if entity.pos.y < top_of_screen:
        entity.pos.y = top_of_screen
    if entity.pos.y > (window_y - entity.size.y):
        entity.pos.y = window_y - entity.size.y

def build_velocity(directions:dict, speed=1):
    vel = Vec2d(0,0)
    if directions['up']:
        vel.y -= speed
    if directions['down']:
        vel.y += speed
    if directions['left']:
        vel.x -= speed
    if directions['right']:
        vel.x += speed
    return vel

def game():
    global difficulty
    pygame.init()

    screen = pygame.display.set_mode((window_x, window_y))
    pygame.display.set_caption("UFO Rush")

    # Load the background image
    background_image = pygame.image.load(os.path.join(asset_folder, "night_sky.jpg"))
    background_image = pygame.transform.scale(background_image, (window_x, window_y))

    player = Entity(
        Vec2d(50, window_y // 2),
        player_size,
        Vec2d(0,0),
        "blue_ufo.png"
    )
    
    # Scoring system and timer
    score = 0
    game_timer = 5 * 60 * 1000  # game duration is five minutes (in milliseconds)
    start_time = pygame.time.get_ticks()
    score_inc = 100

    query_timer = 300

    spawn_timer = 200

    player_current_speed = 10
    
    score_bar = pygame.surface.Surface((window_x, top_of_screen))

    active_pipe_count = 3

    obstacles = []
    for i in range(active_pipe_count):
        obstacle_parts = createObstacle(Vec2d(window_x * (i + 1)//2, window_y // 2 + i * 50))
        obstacles.extend(obstacle_parts)

    # Font to display score and other text
    score_font = pygame.font.SysFont(None, 36)
    game_over_font = pygame.font.SysFont(None, 72)

    paused = True

    # Mode corresponding to expression, 1 is easy, 2 is medium, 3 is hard
    mode = 3
    clock = pygame.time.Clock()
    player_directions = {
        'left':False,
        'right':False,
        'up':False,
        'down':False
    }
    changes = ['Slow Down', 'No Change', 'Speed Up']
    last_change = "No Change"
    # Game loop
    while True:
        # Check for player input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    paused = not paused
                if event.key == pygame.K_LEFT:
                    player_directions["left"] = True
                if event.key == pygame.K_RIGHT:
                    player_directions["right"] = True
                if event.key == pygame.K_DOWN:
                    player_directions["down"] = True
                if event.key == pygame.K_UP:
                    player_directions["up"] = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    player_directions["left"] = False
                if event.key == pygame.K_RIGHT:
                    player_directions["right"] = False
                if event.key == pygame.K_DOWN:
                    player_directions["down"] = False
                if event.key == pygame.K_UP:
                    player_directions["up"] = False
        
        if paused:
            screen.blit(game_over_font.render(f"PAUSED", True, (255,255,255)), (window_x//2-100,window_y//2-50))
            pygame.display.update()
            continue
        # Game Logic
        # Emotion -> change
        if query_timer <= 0:
            emotion_counts = config.counts.copy()
            config.counts[0] = 0
            config.counts[1] = 0
            config.counts[2] = 0
            
            argmax_v = emotion_counts.index(max(emotion_counts))
            if emotion_counts[0] > 30:
                argmax_v = 0
            if argmax_v == 0:
                difficulty = max(difficulty - 1, 0)
            elif argmax_v == 1:
                pass
            else:
                difficulty = min(difficulty + 1, 7)
            last_change = changes[argmax_v]
            query_timer = 600

        # Update Obstacles
        for obstacle in obstacles.copy(): # Need copy else double update
            obstacle.update()
            if (obstacle.pos.x + obstacle.size.x) < 0:
                obstacles.remove(obstacle)
        
        if spawn_timer <= 0:
            obstacles.extend(createObstacle(Vec2d(window_x + 100, random.randint(200, 600))))
            spawn_timer = random.randint(80 - 10 * difficulty, 120 - 12 * difficulty) # Change this with difficulty

        # Update player
        player.velocity = build_velocity(player_directions, player_current_speed)
        player.update()
        hits = pygame.sprite.spritecollide(player, obstacles, False)
        
        for hit in hits:
            if isinstance(hit, ScoreObject):
                obstacles.remove(hit)
                score += score_inc
                active_pipe_count -= 1
            elif isinstance(hit, Obstacle):
                pleft, pright, ptop, pbottom = player.getBounds()
                cleft, cright, ctop, cbottom = hit.getBounds()
                hori_col = min(abs(cright - pleft), abs(cleft - pright))
                vert_col = min(abs(ctop - pbottom), abs(cbottom - ptop))
                if hori_col < vert_col:
                    if pleft < cleft and pright > cleft:
                        player.setRight(cleft)
                    elif pleft < cright and pright > cright:
                        player.setLeft(cright)
                    
                    # if ptop < cbottom and pbottom > cbottom:
                    #     player.setTop(cbottom)
                    # elif ptop < ctop and pbottom > ctop:
                    #     player.setBottom(ctop)
                else:
                    if ptop < cbottom and pbottom > cbottom:
                        player.setTop(cbottom)
                    elif ptop < ctop and pbottom > ctop:
                        player.setBottom(ctop)
                    
                    # if pleft < cleft and pright > cleft:
                    #     player.setRight(cleft)
                    # elif pleft < cright and pright > cright:
                    #     player.setLeft(cright)

        # Force player back onto screen
        bound_to_screen(player)

        # Draw to screen
        # Background
        screen.blit(background_image, (0, 0))  # Draw in the top-left corner (0, 0)
        
        # Player
        screen.blit(player.image, player.pos.getTuple())
        
        # Obstacles
        for obstacle in obstacles:
            if isinstance(obstacle, ScoreObject):
                continue
            screen.blit(obstacle.image, obstacle.pos.getTuple())

        # Menu
        score_bar.fill((0,0,0))
        score_bar.blit(score_font.render(f"Score: {str(score).zfill(6)}", True, (255,255,255)), (0,0))
        score_bar.blit(score_font.render(f"Emotional Response: {last_change}", True, (255,255,255)), (200,0))
        score_bar.blit(score_font.render(f"Next Detect: {query_timer/60:.2f}", True, (255,255,255)), (650,0))
        screen.blit(score_bar,(0,0))

        spawn_timer -= 1
        if config.initialized:
            query_timer -= 1
        
        pygame.display.update()
        clock.tick(60)

if __name__ == '__main__':
    game()