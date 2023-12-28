import sys
from PySide6.QtWidgets import *
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtCore import Qt


class SigVerifWindow(QWidget):
    def __init__(self, model) -> None:
        super().__init__()
        self.file_name = None
        self.InitGUI()
        self.LoadGUI()
        self.RegisterSignals()
        self.model = model

    def InitGUI(self) -> None:
        self.setGeometry(100,100,500,500)
        self.setStyleSheet("background-color: #9EC8B9") 
        self.v_box_layout = QVBoxLayout()
        self.h_box_layout = QHBoxLayout()
        self.h_box_layout_for_image = QHBoxLayout()
        self.h_box_layout_for_import = QHBoxLayout()
        self.h_box_layout_for_verify = QHBoxLayout()


        self.intro_label = QLabel("Welcome!\nPlease enter signature to verify")
        self.intro_label.setStyleSheet("color:white; font-size: 20px ;" "font:sans-serif ;font-weight:bold;")
        self.signature_placeholder = QLabel()
        self.signature_placeholder.setLayout(self.h_box_layout_for_image)
        
        self.signature_picture = QPixmap()
        #self.signature_picture_widget = QWidget()
        self.import_section = QWidget()
        self.import_section.setLayout(self.h_box_layout_for_import)
        self.import_signature_button = QPushButton("import signature")
        self.import_signature_button.setStyleSheet("color:white; background-color: #1B4242;border-radius:5px;height:30px ;font-size: 13px ;" "font:sans-serif ;" "font-weight:bold;  ") 
        
        self.verif_section = QWidget()
        self.verif_section.setLayout(self.h_box_layout_for_verify)
        self.verif_button = QPushButton("verify")
        self.verif_button.setEnabled(False)
        self.verif_button.setStyleSheet("""
                                        QPushButton{
                                                    color: white; 
                                                    background-color: #1B4242;
                                                    border-radius: 5px;
                                                    height: 30px;
                                                    font-size: 13px;
                                                    font: sans-serif ;
                                                    font-weight: bold;
                                                    } 
                                        QPushButton:disabled{
                                                    color:grey;
                                                    }
                                        """) 
        self.verif_result = QLabel("hereeee")
        self.setLayout(self.v_box_layout)
        
    def LoadGUI(self) -> None:
        self.intro_label.setAlignment(Qt.AlignCenter)
        self.intro_label.setFont(QFont("Ariel", 16))
        
        '''This area is responsible for the main layout of the app'''
        self.v_box_layout.addWidget(self.intro_label)
        self.signature_placeholder.setPixmap(self.signature_picture)
        self.v_box_layout.addWidget(self.signature_placeholder)
        self.v_box_layout.addWidget(self.import_section)

        '''This area is responsible for the verification layout'''
        self.h_box_layout.addWidget(self.verif_button)
        # self.verif_result.setText("") 
        self.v_box_layout.addWidget(self.verif_section)

        '''This area is responsible to adjust size of button'''
        self.h_box_layout_for_import.addSpacing(self.h_box_layout_for_import.spacing() + 5)
        self.h_box_layout_for_import.addWidget(self.import_signature_button)
        self.h_box_layout_for_import.addSpacing(self.h_box_layout_for_import.spacing() + 300)

        self.h_box_layout_for_verify.addSpacing(self.h_box_layout_for_verify.spacing() + 5)
        self.h_box_layout_for_verify.addWidget(self.verif_button)
        self.h_box_layout_for_verify.addWidget(self.verif_result)
        self.h_box_layout_for_verify.addSpacing(self.h_box_layout_for_verify.spacing() + 300)

    def RegisterSignals(self) -> None:
        self.import_signature_button.clicked.connect(self.importSignalHandler)
        self.verif_button.clicked.connect(self.verifSignalHandler)

    def verifSignalHandler(self) -> None:
        self.verif_button.setDisabled(True)
        self.verif_result.setText("Loading...")
        result = self.model.predict(self.file_name)
        if(result):
            self.verif_result.setText("Real")
        else:
            self.verif_result.setText("Fake")
        self.verif_button.setEnabled(True)



    def importSignalHandler(self) -> None:
        file_dialog = QFileDialog()
        #file_dialog.setNameFilter()
        file_name, _ = file_dialog.getOpenFileName(self, "Select picture", "", "(*.jpeg *.jpg *.png *.jpeg *.SVG *.TIF *.TIFF *.RAW *.WebP *.BMP)")
        
        if  file_name:
            signature_picture=QPixmap (file_name)
            if (signature_picture.isNull()): 
                print("Error loading image!")
            else:
                self.verif_button.setEnabled(True)
                self.signature_placeholder.setPixmap(signature_picture.scaled(400,200))
                self.file_name = file_name
                    
        