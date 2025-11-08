
import os
import threading
import time
import random
import pickle
from collections import deque

import numpy as np
import chess
import chess.pgn

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "chess_model.pt"
REPLAY_PATH = "replay_buffer.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BOARD_SHAPE = (8, 8, 12)
LR = 3e-4
BATCH_SIZE = 64
REPLAY_CAPACITY = 50000
SELFPLAY_EPISODES_PER_UPDATE = 8

# -----------------------------
# Board helpers
# -----------------------------
PIECE_TO_PLANE = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def board_to_tensor(board: chess.Board):
    tensor = np.zeros(BOARD_SHAPE, dtype=np.float32)
    for sq, piece in board.piece_map().items():
        row, col = divmod(sq, 8)
        r = 7 - row
        c = col
        plane = PIECE_TO_PLANE[piece.piece_type]
        if piece.color == chess.WHITE:
            tensor[r, c, plane] = 1.0
        else:
            tensor[r, c, plane + 6] = 1.0
    return tensor


def legal_moves_mask(board: chess.Board):
    mask = np.zeros(4096, dtype=np.float32)
    for move in board.legal_moves:
        idx = move.from_square * 64 + move.to_square
        mask[idx] = 1.0
    return mask

# -----------------------------
# Model
# -----------------------------
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)

        self.policy_conv = nn.Conv2d(128, 32, 1)
        self.policy_fc = nn.Linear(32 * 8 * 8, 4096)

        self.value_conv = nn.Conv2d(128, 16, 1)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        p = F.relu(self.policy_conv(x))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = F.relu(self.value_conv(x))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v.squeeze(-1)

# -----------------------------
# Replay buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAPACITY):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push_game(self, game_examples):
        for ex in game_examples:
            self.buffer.append(ex)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return batch

    def __len__(self):
        return len(self.buffer)

# -----------------------------
# Trainer
# -----------------------------
class Trainer:
    def __init__(self, model: ChessNet, replay: ReplayBuffer):
        self.model = model.to(DEVICE)
        self.replay = replay
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

    def select_move(self, board: chess.Board, temperature=1.0):
        self.model.eval()
        with torch.no_grad():
            state = board_to_tensor(board)
            state_t = torch.tensor(state).permute(2,0,1).unsqueeze(0).to(DEVICE)
            logits, value = self.model(state_t)
            logits = logits.cpu().numpy().flatten()
            mask = legal_moves_mask(board)
            logits = logits - 1e9 * (1 - mask)
            probs = np.exp(logits / max(1e-8, temperature))
            probs = probs * mask
            if probs.sum() == 0:
                moves = list(board.legal_moves)
                move = random.choice(moves)
                idx = move.from_square*64 + move.to_square
                return move, idx
            probs = probs / probs.sum()
            idx = np.random.choice(len(probs), p=probs)
            from_sq = idx // 64
            to_sq = idx % 64
            move = chess.Move(from_sq, to_sq)
            if move not in board.legal_moves:
                moves = list(board.legal_moves)
                move = random.choice(moves)
                idx = move.from_square*64 + move.to_square
            return move, idx

    def self_play_episode(self, temperature=1.0, max_moves=400):
        board = chess.Board()
        examples = []
        while not board.is_game_over() and board.fullmove_number < max_moves:
            move, idx = self.select_move(board, temperature=temperature)
            state = board_to_tensor(board)
            mask = legal_moves_mask(board)
            examples.append((state, mask, idx, None))
            board.push(move)
        result = board.result()
        if result == '1-0': outcome = 1.0
        elif result == '0-1': outcome = -1.0
        else: outcome = 0.0
        labeled = []
        for i, (s,m,a,_) in enumerate(examples):
            player = 1.0 if i % 2 == 0 else -1.0
            labeled_outcome = outcome * player
            labeled.append((s,m,a,labeled_outcome))
        return labeled, result

    def train_step(self, batch):
        states = np.stack([b[0] for b in batch])
        masks = np.stack([b[1] for b in batch])
        actions = np.array([b[2] for b in batch])
        outcomes = np.array([b[3] for b in batch], dtype=np.float32)

        states_t = torch.tensor(states).permute(0,3,1,2).to(DEVICE)
        actions_t = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        outcomes_t = torch.tensor(outcomes, dtype=torch.float32).to(DEVICE)

        logits, values = self.model(states_t)
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs[torch.arange(len(actions_t)), actions_t]
        advantage = outcomes_t - values.detach()
        policy_loss = - (selected_log_probs * advantage).mean()
        value_loss = F.mse_loss(values, outcomes_t)
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
        self.optimizer.step()
        return float(loss.detach().cpu().item()), float(policy_loss.detach().cpu().item()), float(value_loss.detach().cpu().item())

    def update_from_selfplay(self, episodes=SELFPLAY_EPISODES_PER_UPDATE):
        for _ in range(episodes):
            examples, result = self.self_play_episode()
            self.replay.push_game(examples)
        if len(self.replay) < BATCH_SIZE:
            return None
        stats = []
        for _ in range(10):
            batch = self.replay.sample(BATCH_SIZE)
            stats.append(self.train_step(batch))
        return stats

# -----------------------------
# Utilities to parse PGN / SAN move lists and convert to examples
# -----------------------------

def parse_pgn_file(path):
    games = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            games.append(game)
    return games


def game_to_examples_from_pgn(game):
    board = game.board()
    examples = []
    moves = list(game.mainline_moves())
    for mv in moves:
        state = board_to_tensor(board)
        mask = legal_moves_mask(board)
        idx = mv.from_square*64 + mv.to_square
        examples.append((state, mask, idx, None))
        board.push(mv)
    # Attempt to read result from headers
    result = game.headers.get('Result', '*')
    if result == '1-0': outcome = 1.0
    elif result == '0-1': outcome = -1.0
    elif result == '1/2-1/2': outcome = 0.0
    else: outcome = None
    labeled = []
    if outcome is not None:
        for i, (s,m,a,_) in enumerate(examples):
            player = 1.0 if i % 2 == 0 else -1.0
            labeled_outcome = outcome * player
            labeled.append((s,m,a,labeled_outcome))
    else:
        # leave outcomes None so UI can ask user later
        labeled = examples
    return labeled, result


def parse_san_list(san_text, result=None):
    # san_text: comma or whitespace separated SAN moves like "e4 e5 Nf3"
    tokens = [t.strip() for t in san_text.replace(',', ' ').split() if t.strip()]
    board = chess.Board()
    examples = []
    for tok in tokens:
        try:
            mv = board.parse_san(tok)
        except Exception:
            # skip invalid SAN
            break
        state = board_to_tensor(board)
        mask = legal_moves_mask(board)
        idx = mv.from_square*64 + mv.to_square
        examples.append((state, mask, idx, None))
        board.push(mv)
    if result is None:
        labeled = examples
        res_str = '*'
    else:
        if result == '1-0': outcome = 1.0
        elif result == '0-1': outcome = -1.0
        else: outcome = 0.0
        labeled = []
        for i, (s,m,a,_) in enumerate(examples):
            player = 1.0 if i % 2 == 0 else -1.0
            labeled_outcome = outcome * player
            labeled.append((s,m,a,labeled_outcome))
        res_str = result
    return labeled, res_str

# -----------------------------
# GUI
# -----------------------------
class ChessGUI:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
        self.replay = trainer.replay
        self.board = chess.Board()
        self.square_size = 64
        self.root = tk.Tk()
        self.root.title('Chess Bot — Teach & Play')

        # canvas
        self.canvas = tk.Canvas(self.root, width=8*self.square_size, height=8*self.square_size)
        self.canvas.grid(row=0, column=0, rowspan=8)
        self.canvas.bind('<Button-1>', self.on_click)

        # controls
        control_frame = tk.Frame(self.root)
        control_frame.grid(row=0, column=1, sticky='nw')

        tk.Button(control_frame, text='Reset', width=15, command=self.reset).pack(pady=2)
        tk.Button(control_frame, text='Load Model', width=15, command=self.load_model).pack(pady=2)
        tk.Button(control_frame, text='Save Model', width=15, command=self.save_model).pack(pady=2)
        tk.Button(control_frame, text='Self-play (25)', width=15, command=self.self_play_background).pack(pady=2)
        tk.Button(control_frame, text='Train (10 iters)', width=15, command=self.train_background).pack(pady=2)
        tk.Button(control_frame, text='Load PGN', width=15, command=self.load_pgn).pack(pady=2)
        tk.Button(control_frame, text='Paste Moves', width=15, command=self.paste_moves_dialog).pack(pady=2)
        tk.Button(control_frame, text='Teach from Moves', width=15, command=self.teach_from_textbox).pack(pady=2)

        tk.Label(control_frame, text='Temperature:').pack()
        self.temp_var = tk.DoubleVar(value=0.7)
        tk.Scale(control_frame, from_=0.1, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, variable=self.temp_var).pack()

        tk.Label(control_frame, text='Status:').pack()
        self.status_var = tk.StringVar(value='Ready')
        tk.Label(control_frame, textvariable=self.status_var, wraplength=200, justify='left').pack()

        # textbox to paste moves
        tk.Label(control_frame, text='Moves / PGN text:').pack()
        self.moves_text = tk.Text(control_frame, width=30, height=10)
        self.moves_text.pack()

        # internal UI state
        self.selected_square = None
        self.last_move = None
        self.human_examples = []

        # load replay if present
        if os.path.exists(REPLAY_PATH):
            try:
                with open(REPLAY_PATH, 'rb') as f:
                    self.replay.buffer = pickle.load(f)
                    self.status('Loaded replay buffer, size=%d' % len(self.replay))
            except Exception:
                pass

        # load model if present
        if os.path.exists(MODEL_PATH):
            try:
                self.trainer.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
                self.status('Loaded model weights')
            except Exception:
                pass

        self.draw_board()

    def status(self, text):
        self.status_var.set(text)
        self.root.update_idletasks()

    def draw_board(self):
        self.canvas.delete('all')
        colors = ['#EEEED2', '#769656']
        for row in range(8):
            for col in range(8):
                x1 = col*self.square_size
                y1 = row*self.square_size
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                color = colors[(row+col)%2]
                self.canvas.create_rectangle(x1,y1,x2,y2, fill=color, outline='')
        # highlight selected
        if self.selected_square is not None:
            r, c = divmod(self.selected_square, 8)
            rr = 7 - r
            x1 = c*self.square_size
            y1 = rr*self.square_size
            self.canvas.create_rectangle(x1,y1,x1+self.square_size,y1+self.square_size, outline='blue', width=3)
        # draw pieces
        symbols = { 'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔', 'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚' }
        for sq,piece in self.board.piece_map().items():
            row,col = divmod(sq,8)
            x = col*self.square_size + self.square_size//2
            y = (7-row)*self.square_size + self.square_size//2
            self.canvas.create_text(x,y, text=symbols[piece.symbol()], font=('Arial', 36))
        # show last move
        if self.last_move is not None:
            from_sq, to_sq = self.last_move
            for s in [from_sq, to_sq]:
                r,c = divmod(s,8)
                rr = 7 - r
                x1 = c*self.square_size
                y1 = rr*self.square_size
                self.canvas.create_rectangle(x1,y1,x1+self.square_size,y1+self.square_size, outline='red', width=2)

    def on_click(self, event):
        if self.board.is_game_over():
            messagebox.showinfo('Game Over', f'Result: {self.board.result()}')
            return
        col = event.x // self.square_size
        row = 7 - (event.y // self.square_size)
        sq = chess.square(col, row)
        if self.selected_square is None:
            if self.board.piece_at(sq) and self.board.piece_at(sq).color == chess.WHITE:
                self.selected_square = sq
        else:
            move = chess.Move(self.selected_square, sq)
            if move in self.board.legal_moves:
                # record example for player's move
                s = board_to_tensor(self.board)
                mask = legal_moves_mask(self.board)
                a = move.from_square*64 + move.to_square
                self.human_examples.append((s,mask,a,None))
                self.board.push(move)
                self.last_move = (move.from_square, move.to_square)
                self.selected_square = None
                self.draw_board()
                if self.board.is_game_over():
                    self.handle_game_end()
                    return
                # bot move
                self.root.after(100, self.bot_move)
            else:
                # invalid move or deselect
                self.selected_square = None
        self.draw_board()

    def bot_move(self):
        temp = float(self.temp_var.get())
        move, idx = self.trainer.select_move(self.board, temperature=temp)
        s = board_to_tensor(self.board)
        mask = legal_moves_mask(self.board)
        self.board.push(move)
        self.human_examples.append((s, mask, idx, None))
        self.last_move = (move.from_square, move.to_square)
        self.draw_board()
        if self.board.is_game_over():
            self.handle_game_end()

    def handle_game_end(self):
        result = self.board.result()
        if result == '1-0': outcome = -1.0; msg = 'You Win!'
        elif result == '0-1': outcome = 1.0; msg = 'Bot Wins!'
        else: outcome = 0.0; msg = 'Draw.'
        messagebox.showinfo('Game Over', msg)
        labeled = []
        for i, (s,m,a,_) in enumerate(self.human_examples):
            player = 1.0 if i % 2 == 0 else -1.0
            labeled_outcome = outcome * player
            labeled.append((s,m,a,labeled_outcome))
        self.replay.push_game(labeled)
        self.human_examples.clear()
        # save replay automatically
        try:
            with open(REPLAY_PATH, 'wb') as f:
                pickle.dump(self.replay.buffer, f)
            self.status('Game saved to replay buffer (size=%d)' % len(self.replay))
        except Exception as e:
            self.status('Could not save replay: %s' % e)

    def reset(self):
        self.board = chess.Board()
        self.selected_square = None
        self.last_move = None
        self.human_examples.clear()
        self.draw_board()

    def save_model(self):
        torch.save(self.trainer.model.state_dict(), MODEL_PATH)
        self.status('Model saved to %s' % MODEL_PATH)

    def load_model(self):
        if os.path.exists(MODEL_PATH):
            self.trainer.model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            self.status('Model loaded')
        else:
            self.status('No model file found')

    # Background tasks
    def run_in_thread(self, target, *args):
        t = threading.Thread(target=target, args=args, daemon=True)
        t.start()

    def train_background(self):
        self.run_in_thread(self.train_worker)

    def train_worker(self):
        self.status('Training... running self-play + updates')
        stats = self.trainer.update_from_selfplay(episodes=8)
        if stats is None:
            self.status('Not enough data in replay buffer to train (need more examples).')
            return
        avg_losses = [sum(s[i] for s in stats)/len(stats) for i in range(3)]
        self.status(f'Training done. avg_loss={avg_losses[0]:.4f}')
        # save replay and model
        with open(REPLAY_PATH, 'wb') as f:
            pickle.dump(self.replay.buffer, f)
        torch.save(self.trainer.model.state_dict(), MODEL_PATH)

    def self_play_background(self):
        self.run_in_thread(self.self_play_worker)

    def self_play_worker(self):
        self.status('Running self-play (25 eps)')
        for i in range(25):
            examples, result = self.trainer.self_play_episode()
            self.replay.push_game(examples)
            self.status(f'Self-play {i+1}/25 done; result {result}; replay={len(self.replay)}')
        with open(REPLAY_PATH, 'wb') as f:
            pickle.dump(self.replay.buffer, f)
        self.status('Self-play finished and saved to replay')

    # Teach from PGN / SAN moves
    def load_pgn(self):
        path = filedialog.askopenfilename(filetypes=[('PGN files','*.pgn'), ('All files','*.*')])
        if not path: return
        games = parse_pgn_file(path)
        count = 0
        for g in games:
            examples, res = game_to_examples_from_pgn(g)
            # if examples have labeled outcomes (res known) they will be labeled already
            self.replay.push_game(examples)
            count += 1
        with open(REPLAY_PATH, 'wb') as f:
            pickle.dump(self.replay.buffer, f)
        self.status(f'Loaded {count} games from PGN into replay (size={len(self.replay)})')

    def paste_moves_dialog(self):
        # helper: user can paste SAN moves into textbox
        self.moves_text.delete('1.0', tk.END)
        self.status('Paste SAN moves into the box and click Teach from Moves')

    def teach_from_textbox(self):
        txt = self.moves_text.get('1.0', tk.END).strip()
        if not txt:
            self.status('No moves to teach from')
            return
        # ask for result
        res = simpledialog.askstring('Game result', 'Enter result (1-0, 0-1, 1/2-1/2) or leave blank if unknown:')
        examples, res_str = parse_san_list(txt, result=res)
        if len(examples) == 0:
            self.status('No valid moves parsed')
            return
        self.replay.push_game(examples)
        with open(REPLAY_PATH, 'wb') as f:
            pickle.dump(self.replay.buffer, f)
        self.status(f'Taught from moves; added {len(examples)} positions to replay (size={len(self.replay)})')

    def mainloop(self):
        self.root.mainloop()

# -----------------------------
# Main
# -----------------------------

def main():
    model = ChessNet()
    replay = ReplayBuffer()
    # load replay if exists
    if os.path.exists(REPLAY_PATH):
        try:
            with open(REPLAY_PATH, 'rb') as f:
                replay.buffer = pickle.load(f)
        except Exception:
            pass
    trainer = Trainer(model, replay)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            print('Loaded model')
        except Exception:
            pass

    gui = ChessGUI(trainer)
    gui.mainloop()

if __name__ == '__main__':
    main()
