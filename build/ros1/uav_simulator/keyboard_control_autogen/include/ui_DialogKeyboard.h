/********************************************************************************
** Form generated from reading UI file 'DialogKeyboard.ui'
**
** Created by: Qt User Interface Compiler version 5.12.8
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOGKEYBOARD_H
#define UI_DIALOGKEYBOARD_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QDialog>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>

QT_BEGIN_NAMESPACE

class Ui_DialogKeyboard
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;

    void setupUi(QDialog *DialogKeyboard)
    {
        if (DialogKeyboard->objectName().isEmpty())
            DialogKeyboard->setObjectName(QString::fromUtf8("DialogKeyboard"));
        DialogKeyboard->resize(360, 229);
        DialogKeyboard->setMaximumSize(QSize(701, 411));
        verticalLayout = new QVBoxLayout(DialogKeyboard);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        label = new QLabel(DialogKeyboard);
        label->setObjectName(QString::fromUtf8("label"));
        label->setPixmap(QPixmap(QString::fromUtf8(":/droneKeyboard/droneKeyboard.png")));
        label->setScaledContents(true);

        verticalLayout->addWidget(label);


        retranslateUi(DialogKeyboard);

        QMetaObject::connectSlotsByName(DialogKeyboard);
    } // setupUi

    void retranslateUi(QDialog *DialogKeyboard)
    {
        DialogKeyboard->setWindowTitle(QApplication::translate("DialogKeyboard", "Keyboard controller", nullptr));
        label->setText(QString());
    } // retranslateUi

};

namespace Ui {
    class DialogKeyboard: public Ui_DialogKeyboard {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOGKEYBOARD_H
