from typing import Dict, Optional, Union
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior, RecycleKVIDsDataViewBehavior
from kivy.uix.label import Label
from kivy.properties import BooleanProperty, ObjectProperty, NumericProperty, StringProperty
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput

from kivy.clock import Clock
from random import random


Builder.load_string('''
<TypeButton@Button>:
    size_hint_y: None
    height: 45
    background_color: 0.0, 0.5, 1.0, 1.0

<TypeDropDown>:
    TypeButton:
        text: 'int16'
        on_release: root.select('int16')

    TypeButton:
        text: 'int32'
        on_release: root.select('int32')

    TypeButton:
        text: 'int64'
        on_release: root.select('int64')

    TypeButton:
        text: 'uint16'
        on_release: root.select('uint16')

    TypeButton:
        text: 'uint32'
        on_release: root.select('uint32')

    TypeButton:
        text: 'uint64'
        on_release: root.select('uint64')

    TypeButton:
        text: 'float32'
        on_release: root.select('float32')

    TypeButton:
        text: 'float64'
        on_release: root.select('float64')

<SelectableLabel>:
    offset_label: offset_label
    value_input: value_input
    formatter_button: formatter_button
    size_hint_y: None
    height: 45

    canvas.before:
        Color:
            rgba: (.0, 0.9, .1, .3) if self.selected else (0, 0, 0, 1)
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        id: offset_label
        text: str(root.offset)

    TextInput:
        id: value_input
        text: str(root.value)
        disabled: not root.selected
        multiline: False
        input_type: "number"
        input_filter: "float"

    Button:
        id: formatter_button
        text: root.formatter


<RV>:
    viewclass: 'SelectableLabel'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
''')


class TypeDropDown(DropDown):
    pass


class SelectableRecycleBoxLayout(FocusBehavior,
                                 LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    pass


class SelectableLabel(RecycleDataViewBehavior, BoxLayout):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    offset = NumericProperty()
    value = NumericProperty(1)
    formatter = StringProperty('uint16')

    offset_label: Label = ObjectProperty()
    value_input: TextInput = ObjectProperty()
    formatter_button: Button = ObjectProperty()
    formatter_dropdown: Optional[TypeDropDown] = ObjectProperty()

    def __init__(self, *args, **kwargs) -> None:
        dropdown = self.formatter_dropdown = TypeDropDown()
        dropdown.bind(on_select=self.change_formatter)
        super().__init__(*args, **kwargs)
        self.formatter_button.bind(on_release=dropdown.open)
        self.value_input.bind(on_text_validate=self.change_value)

    def change_value(self, instance):
        if instance.text:
            self.update_data(False, value=float(instance.text))

    def change_formatter(self, instance, value):
        self.update_data(formatter=value)

    @property
    def recycleview(self) -> RecycleView:
        return self.parent.parent

    def update_data(self, dispatch: bool = True, **attrs: Dict[str, Union[float, int, str]]):
        if dispatch:
            self.recycleview.data[self.index] = {
                **self.recycleview.data[self.index].copy(),
                **attrs
            }
        else:
            self.recycleview.data[self.index].update(attrs)

    def refresh_view_attrs(self, rv: RecycleView, index: int, data: Dict):
        self.index = index
        return super(SelectableLabel, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        if super(SelectableLabel, self).on_touch_down(touch):
            return not self.selected
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        rv.data[index]['selected'] = self.selected = is_selected


class RV(RecycleView):
    def __init__(self, **kwargs):
        super(RV, self).__init__(**kwargs)
        self.data_model.bind(on_data_changed=self.change_data)
        self.data = list(self.generate_data())
        # Clock.schedule_interval(self.update_values, 1)

    def generate_data(self, count=5):
        for i in range(count):
            yield dict(offset=i, formatter='uint16')

    def change_data(self, model, **kwargs):
        print(model, kwargs)

    def update_values(self, dt):
        self.data[0:len(self.data)] = [
            {**kwargs, 'value': random()} for kwargs in self.data
        ]


class TestApp(App):
    def build(self):
        return RV()


if __name__ == '__main__':
    TestApp().run()
