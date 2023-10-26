import sys
import os.path
import pickle

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QGridLayout, QHBoxLayout, QVBoxLayout,
    QRadioButton, QPushButton, QMessageBox,
)

from tictac_rl.tic_tac_toe_environment import TicTacToeEnvironment


class MainWindow(QMainWindow):
    def __init__(self, agent_x, agent_o):
        super().__init__()

        layout = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        board_layout = QGridLayout()

        self.agent_x = agent_x
        self.agent_o = agent_o
        self.agent_x.eps = 0
        self.agent_o.eps = 0

        self.play_with = 'x'
        self.env = TicTacToeEnvironment(agent_o)

        self.xbutton = QRadioButton('x')
        self.xbutton.setChecked(True)
        self.xbutton.toggled.connect(lambda: self.btnstate(self.xbutton))
        buttons_layout.addWidget(self.xbutton)
            
        self.obutton = QRadioButton('o')
        self.obutton.toggled.connect(lambda: self.btnstate(self.obutton))

        buttons_layout.addWidget(self.obutton)
        layout.addLayout(buttons_layout)
        
        self.buttons = []
        for i in range(3):
            for j in range(3):
                btn = QPushButton()
                self.buttons.append(btn)
                board_layout.addWidget(btn, i, j)
                btn.clicked.connect(lambda checked=True, b=btn: self.btn_clicked(b))
                btn.i = i
                btn.j = j
                board_layout.addWidget(btn)
        layout.addLayout(board_layout)

        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)
        self.setWindowTitle('TicTac RL')

        self.msgbox = QMessageBox()
    
    def reset_player(self, symbol):
        self.play_with = symbol
        self.env = TicTacToeEnvironment(self.agent_o if symbol == 'x' else self.agent_x)
        self.update_board(self.env.board)

    def btnstate(self, b):
        self.reset_player(b.text())
    
    def update_board(self, board):
        for i in range(3):
            for j in range(3):
                self.buttons[3*i + j].setText(board[i, j])
    
    def btn_clicked(self, btn):
        if btn.text():
            return
        btn.setText(self.play_with)
        board, _, winner, tie = self.env.step([btn.i, btn.j])
        
        if winner == self.play_with:
            self.msgbox.setText('Congratulations!')
            self.msgbox.exec()
            self.reset_player('x')
            self.reset_player('o')
            self.reset_player('x')
        if winner == self.env.trained_player.play_with:
            self.msgbox.setText('You lose. Game over')
            self.msgbox.exec()
            self.reset_player('x')
            self.reset_player('o')
            self.reset_player('x')
        if tie:
            self.msgbox.setText("It's a draw")
            self.msgbox.exec()
            self.reset_player('x')
            self.reset_player('o')
            self.reset_player('x')
        
        self.update_board(board)


if __name__ == '__main__':
    
    x_model_path = os.path.join(os.path.dirname(__file__), 'models', 'trained_x_gui.pkl')
    o_model_path = os.path.join(os.path.dirname(__file__), 'models', 'trained_o_gui.pkl')
    assert os.path.isfile(x_model_path) and os.path.isfile(o_model_path), 'Trained model files not found'
    
    with open(x_model_path, 'rb') as f:
        agent_x = pickle.load(f)
    with open(o_model_path, 'rb') as f:
        agent_o = pickle.load(f)

    app = QApplication([])
    w = MainWindow(agent_x, agent_o)
    w.show()
    
    sys.exit(app.exec())

