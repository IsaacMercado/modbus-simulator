from kivy.lang import Builder
from kivy.properties import StringProperty
from kivymd.app import MDApp

from kivymd.uix.list import OneLineListItem
from kivymd.uix.behaviors.toggle_behavior import MDToggleButton
from kivymd.uix.button import MDFillRoundFlatButton

from kivymd.uix.tab import MDTabsBase
from kivymd.uix.floatlayout import MDFloatLayout

from kivymd.uix.button import MDFlatButton
from kivymd.uix.dialog import MDDialog


KV = '''
<Check@MDCheckbox>:
    group: 'group'
    size_hint: None, None
    size: dp(48), dp(48)

<Tab>:
    MDLabel:
        text: root.content_text
        pos_hint: {"center_x": .5, "center_y": .5}

MDBoxLayout:
    orientation: 'vertical'
    md_bg_color: (0., 0., 0., .25)

    MDBoxLayout:
        size_hint: 1, None
        height: '45sp'

        MDBoxLayout:
            oritentacion: 'vertical'

            MDBoxLayout:
                size_hint: None, 1
                Label:
                    text: "TCP"
                Check:
                    active: True
            
            MDBoxLayout:
                size_hint: None, 1
                Label:
                    text: "Serial"
                Check:
        
        Widget:

        MDBoxLayout:
            id: interface_settings
            orientation: 'vertical' if self.width < self.height else 'horizontal'
            size_hint: None, 1
            width: '200sp'
            padding: '2sp'

            MDLabel:
                text: "Port"

            MDTextField:
                id: txtBox
                text: "5440"
                multiline: False
                mode: "rectangle"

        Widget:

        CustomToggleButton:
            id: start_stop_server
            size_hint: None, 1
            width: '108sp'
            text: 'Start'

    MDSeparator:

    MDBoxLayout:
        orientation: 'vertical' if self.width < self.height else 'horizontal'
        
        MDBoxLayout:
            orientation: 'vertical'
            padding: '2sp'
            size_hint_x: .4

            MDLabel:
                text: "modbus slaves"
                halign: "center"
                size_hint_y: .1

            MDSeparator:

            MDBoxLayout:
                orientation: 'horizontal'
                padding: '2sp'
                size_hint: 1, None
                height: '45sp'
                
                MDFlatButton:
                    text: "Add"
                
                MDFlatButton:
                    text: "Delete"

                MDFlatButton:
                    text: "Enable all"

            MDSeparator:

            MDBoxLayout:
                orientation: 'vertical'
                padding: '2sp'
                size_hint: 1, None
                height: '100sp'

                MDBoxLayout:
                    orientation: "horizontal"
                    
                    MDLabel:
                        text: 'from'
                        size_hint: .25, 1

                    MDTextFieldRound:
                        hint_text: "from"
                        required: True
                        text: '1'

                MDBoxLayout:
                    orientation: "horizontal"
                    
                    MDLabel:
                        text: 'to'
                        size_hint: .25, 1
                
                    MDTextFieldRound:
                        hint_text: "to"
                        required: True
                        text: '1'

                MDBoxLayout:
                    orientation: "horizontal"
                    
                    MDLabel:
                        text: 'count'
                        size_hint: .25, 1

                    MDTextFieldRound:
                        hint_text: "count"
                        required: True
                        text: '1'
            
            MDSeparator:

            ScrollView:
                MDList:
                    id: container

        MDBoxLayout:
            orientation: 'vertical'
            padding: '2sp'

            MDBoxLayout:
                orientation: 'horizontal'
                padding: '2sp'
                size_hint: 1, None
                height: '45sp'

                MDLabel:
                    text: "Count"

                Widget:

                MDTextFieldRound:
                    text: "10"

            MDSeparator:

            MDFlatButton:
                padding: '2sp'
                size_hint: 1, None
                height: '45sp'
                text: "add"
            
            MDSeparator:

            MDTabs:

                Tab:
                    title: "Coils"
                    reference: "coils"
                    content_text: f"This is an example text for {self.title}"

                Tab:
                    title: "Discrete inputs"
                    reference: "discrete_inputs"
                    content_text: f"This is an example text for {self.title}"

                Tab:
                    title: "Input registers"
                    reference: "input_registers"
                    content_text: f"This is an example text for {self.title}"

                Tab:
                    title: "Holding registers"
                    reference: "holding_registers"
                    content_text: f"This is an example text for {self.title}"
    
    MDSeparator:

    MDToolbar:
        title: "Modbus simulator"
        type: "bottom"
        right_action_items: [["play", lambda x: print(x)], ["reload", lambda x: print(x)], ["tools", lambda x: print(x)]]
        mode: "free-end"
'''


class Tab(MDFloatLayout, MDTabsBase):
    content_text: str = StringProperty('')
    reference: str = StringProperty('')
    dialog: MDDialog | None = None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class CustomToggleButton(MDFillRoundFlatButton, MDToggleButton):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_down = self.theme_cls.primary_light


class MainApp(MDApp):
    def build(self):
        return Builder.load_string(KV)

    def on_start(self):
        for i in range(5):
            self.root.ids.container.add_widget(
                OneLineListItem(text=f"Single-line item {i}")
            )


MainApp().run()
