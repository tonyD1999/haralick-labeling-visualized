from pathlib import Path

import numpy as np

from utils import neighborhood_values
from visualizer import HaralickVisualizer


def ccl_8(image, display=False):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
    equil_relationship = {0:[]}
    image = np.pad(image, 1, pad_with)
    image[image > 0] = np.arange(1, np.sum(image > 0) + 1)
    visualizer = HaralickVisualizer(index_image=image,
                                    canvas_size=(512, 512),
                                    output_dir=Path('./output'))


    state = {
        'iter': 0,
        'finished': False,
        'mode': 'forward',
        'pos': (0, 0),
        'step': 0
    }
    state['iter'] += 1
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            state['step'] += 1
            state['pos'] = (i, j)
            current_pixel = image[i][j]
            neighbors_ = [(image[i][j-1], image[i][j-1]),
                         (image[i-1][j], image[i-1][j]),
                         (image[i-1][j-1], image[i-1][j-1]),
                         (image[i-1][j+1], image[i-1][j+1])]
                         
            neighbors = list(filter(lambda x: x[0]!=0, neighbors_))
            if current_pixel == 0:
                continue
            if len(neighbors) > 0:
                label_list = list(map(lambda x: x[1], neighbors))
                image[i][j] = min(label_list)
                for _ in neighbors:
                    equil_relationship[_[1]] = set(list(equil_relationship[_[1]]) + label_list)
            else:
                new_label = list(equil_relationship.keys())[-1] + 1
                equil_relationship[new_label] = []
                image[i][j] = new_label
            if display:
                visualizer.display(state, title='Labeling...', wait=1)

    for key, values in equil_relationship.items():
        for value in values:
            equil_relationship[key] = equil_relationship[key].union(equil_relationship[value])
        
    state['iter'] += 1
    state['mode'] = 'forward'
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            state['step'] += 1
            state['pos'] = (i, j)
            current_pixel = image[i][j]
            if current_pixel == 0:
                continue
            labels = equil_relationship[image[i][j]]
            if len(labels) > 0:
                assign_label = min(labels)
                image[i][j] = assign_label
            if display:    
                visualizer.display(state, title='Labeling...', wait=1)
    state['finished'] = True
    visualizer.display(state, title='RESULT', wait=0)

def ccl_4(image, display=False):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
    equil_relationship = {0:[]}
    image = np.pad(image, 1, pad_with)
    image[image > 0] = np.arange(1, np.sum(image > 0) + 1)
    visualizer = HaralickVisualizer(index_image=image,
                                    canvas_size=(512, 512),
                                    output_dir=Path('./output'))


    state = {
        'iter': 0,
        'finished': False,
        'mode': 'forward',
        'pos': (0, 0),
        'step': 0
    }
    state['iter'] += 1
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            state['step'] += 1
            state['pos'] = (i, j)
            current_pixel = image[i][j]
            neighbors_ = [(image[i][j-1], image[i][j-1]),
                         (image[i-1][j], image[i-1][j])]
                         
            neighbors = list(filter(lambda x: x[0]!=0, neighbors_))
            if current_pixel == 0:
                continue
            if len(neighbors) > 0:
                label_list = list(map(lambda x: x[1], neighbors))
                image[i][j] = min(label_list)
                for _ in neighbors:
                    equil_relationship[_[1]] = set(list(equil_relationship[_[1]]) + label_list)
            else:
                new_label = list(equil_relationship.keys())[-1] + 1
                equil_relationship[new_label] = []
                image[i][j] = new_label
            if display:
                visualizer.display(state, title='Labeling...', wait=1)

    for key, values in equil_relationship.items():
        for value in values:
            equil_relationship[key] = equil_relationship[key].union(equil_relationship[value])
        
    state['iter'] += 1
    state['mode'] = 'forward'
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            state['step'] += 1
            state['pos'] = (i, j)
            current_pixel = image[i][j]
            if current_pixel == 0:
                continue
            labels = equil_relationship[image[i][j]]
            if len(labels) > 0:
                assign_label = min(labels)
                image[i][j] = assign_label
            if display:    
                visualizer.display(state, title='Labeling...', wait=1)
    state['finished'] = True
    visualizer.display(state, title='RESULT', wait=0)

def bfs(image, display=False):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
    image = np.pad(image, 1, pad_with)
    # label = np.zeros(image.shape, dtype=np.int)
    image[image > 0] = np.arange(1, np.sum(image > 0) + 1)
    visualizer = HaralickVisualizer(index_image=image,
                                    canvas_size=(512, 512),
                                    output_dir=Path('./output'))


    state = {
        'iter': 0,
        'finished': False,
        'mode': 'search',
        'pos': (0, 0),
        'step': 0
    }
    visited = np.zeros(image.shape, dtype=np.int)
    count = 1
    state['iter'] += 1
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            current_pixel = image[i][j]
            state['pos'] = (i, j)
            # if display:
            #     visualizer.display(state, title='Labeling...', wait=1)
            if current_pixel == 0 or visited[i][j] == 1:
                continue
            queue = []
            queue.append((i, j))
            # visited[i][j] = 1
            # image[i][j] = count
            while queue:
                current_x, current_y = queue.pop(0)
                state['step'] += 1
                state['pos'] = (current_x, current_y)
                if display:
                    visualizer.display(state, title='Labeling...', wait=1)
                if visited[current_x][current_y] == 1:
                    continue
                visited[current_x][current_y] = 1
                image[current_x][current_y] = count
                neighbors_ = [(image[current_x][current_y-1], (current_x, current_y-1)),
                        (image[current_x-1][current_y], (current_x-1, current_y)),
                        (image[current_x-1][current_y-1], (current_x-1, current_y-1)),
                        (image[current_x-1][current_y+1], (current_x-1, current_y+1)),
                        (image[current_x][current_y+1], (current_x, current_y+1)),
                        (image[current_x+1][current_y], (current_x+1, current_y)),
                        (image[current_x+1][current_y-1], (current_x+1, current_y-1)),
                        (image[current_x+1][current_y+1], (current_x+1, current_y+1))]
                neighbors = list(filter(lambda x: x[0]!=0, neighbors_))
                neighbors = list(map(lambda x: x[1], neighbors))
                for neighbor in neighbors:
                    neighbor_x, neighbor_y = neighbor
                    state['step'] += 1
                    state['pos'] = (neighbor_x, neighbor_y)
                    if visited[neighbor_x][neighbor_y] == 0 and (neighbor_x, neighbor_y) not in queue:
                        queue.append(neighbor)
                    #     visited[neighbor_x][neighbor_y] = 1
                    #     image[neighbor_x][neighbor_y] = count
                    # if display:
                    #     visualizer.display(state, title='Labeling...', wait=1)
            count += 1
    state['finished'] = True
    visualizer.display(state, title='RESULT', wait=0)

def dfs(image, display=False):
    def pad_with(vector, pad_width, iaxis, kwargs):
        pad_value = kwargs.get('padder', 0)
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
    image = np.pad(image, 1, pad_with)
    # label = np.zeros(image.shape, dtype=np.int)
    image[image > 0] = np.arange(1, np.sum(image > 0) + 1)
    visualizer = HaralickVisualizer(index_image=image,
                                    canvas_size=(512, 512),
                                    output_dir=Path('./output'))


    state = {
        'iter': 0,
        'finished': False,
        'mode': 'search',
        'pos': (0, 0),
        'step': 0
    }
    visited = np.zeros(image.shape, dtype=np.int)
    count = 1
    state['iter'] += 1
    for i in range(1, len(image)-1):
        for j in range(1, len(image[i])-1):
            state['pos'] = (i, j)
            # if display:
            #     visualizer.display(state, title='Labeling...', wait=1)
            current_pixel = image[i][j]
            if current_pixel == 0 or visited[i][j] == 1:
                continue
            stack = []
            stack.append((i, j))
            
            while stack:
                current_x, current_y = stack.pop()
                state['pos'] = (current_x, current_y)
                state['step'] += 1
                if display:
                    visualizer.display(state, title='Labeling...', wait=1)

                if visited[current_x][current_y] == 1:
                    continue
                visited[current_x][current_y] = 1
                image[current_x][current_y] = count
                neighbors_ = [(image[current_x][current_y-1], (current_x, current_y-1)),
                        (image[current_x-1][current_y], (current_x-1, current_y)),
                        (image[current_x-1][current_y-1], (current_x-1, current_y-1)),
                        (image[current_x-1][current_y+1], (current_x-1, current_y+1)),
                        (image[current_x][current_y+1], (current_x, current_y+1)),
                        (image[current_x+1][current_y], (current_x+1, current_y)),
                        (image[current_x+1][current_y-1], (current_x+1, current_y-1)),
                        (image[current_x+1][current_y+1], (current_x+1, current_y+1))]
                neighbors = list(filter(lambda x: x[0]!=0, neighbors_))
                neighbors = list(map(lambda x: x[1], neighbors))
                for neighbor in neighbors:
                    neighbor_x, neighbor_y = neighbor
                    if visited[neighbor_x][neighbor_y] == 0 and (neighbor_x, neighbor_y) not in stack:
                        stack.append(neighbor)
                    
            count += 1
    state['finished'] = True
    visualizer.display(state, title='RESULT', wait=0)





    
