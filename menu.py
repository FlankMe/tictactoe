# -*- coding: utf-8 -*-
"""
Created on Sat May 14 21:45:20 2016

@author: Riccardo Rossi
"""

import pygame

def launch(screen):
    font_menu = pygame.font.Font(None, 24)
    
    menu_running = True
    while menu_running: 
        for event in pygame.event.get(): 
            if event.type == pygame.KEYDOWN: 
                if event.key == pygame.K_y:
                    HUMAN_START = True
                    menu_running = False  
                if event.key == pygame.K_n:
                    HUMAN_START = False
                    menu_running = False  
        
        screen.fill(BLACK)        
        
        menu = font_menu.render('WOULD YOU LIKE TO GO FIRST ?', 1, YELLOW)
        text_rect = menu.get_rect()
        text_rect.centery = screen.get_rect().centery
        text_rect.centerx = screen.get_rect().centerx
        screen.blit(menu, text_rect)
        
        menu = font_menu.render('(press \'y\' or \'n\')', 1, YELLOW)
        text_rect = menu.get_rect()
        text_rect.centery = screen.get_rect().centery + 25
        text_rect.centerx = screen.get_rect().centerx
        screen.blit(menu, text_rect)

        pygame.display.flip()
    
    del menu, font_menu
    return(HUMAN_START)
            
# Define some colors
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255,   0,   0)
BLUE  = (  0,   0, 255)
GREEN = (  0, 100,   0)
YELLOW =(255, 250, 205)