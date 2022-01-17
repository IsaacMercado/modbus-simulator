from typing import Any, Dict, Optional, Union
from random import random

from kivy.app import App
from kivy.lang import Builder
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.properties import (
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty
)

from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleDataViewBehavior
from kivy.uix.label import Label
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput


Builder.load_string('''
<RowData>:
    offset_label: loffset
    value_input: ivalue
    formatter_button: bformatter
    size_hint_y: None
    height: 45

    canvas.before:
        Color:
            rgba: (.0, 0.9, .1, .3) if self.selected else (0, 0, 0, 1)
        Rectangle:
            pos: self.pos
            size: self.size

    Label:
        id: loffset
        text: str(root.offset)

    TextInput:
        id: ivalue
        text: str(root.value)
        disabled: not root.selected
        multiline: False
        input_type: "number"
        input_filter: "float"
        on_focus: self.text = str(root.value) # correcting value

    Button:
        id: bformatter
        text: root.formatter

<RV>:
    viewclass: 'RowData'
    SelectableRecycleBoxLayout:
        default_size: None, dp(56)
        default_size_hint: 1, None
        size_hint_y: None
        height: self.minimum_height
        orientation: 'vertical'
''')


dtypes = [
    'int16',
    'int32',
    'int64',
    'uint16',
    'uint32',
    'uint64',
    'float32',
    'float64'
]

Data = Dict[str, Union[int, float, str]]

DEFAULT_VALUE = 1


class TypeDropDown(DropDown):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for dtype in dtypes:
            button = Button(text=dtype, size_hint_y=None, height=45,
                            background_color=[0.0, 0.5, 1.0, 1.0])
            button.bind(on_release=lambda instance: self.select(instance.text))
            self.add_widget(button)


class SelectableRecycleBoxLayout(FocusBehavior,
                                 LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    pass


class RowData(RecycleDataViewBehavior, BoxLayout):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    offset = NumericProperty()
    value = NumericProperty(DEFAULT_VALUE)
    formatter = StringProperty('uint16')

    offset_label: Label = ObjectProperty()
    value_input: TextInput = ObjectProperty()
    formatter_button: Optional[Button] = ObjectProperty()
    formatter_dropdown: Optional[DropDown] = ObjectProperty()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.formatter_dropdown = TypeDropDown()
        self.formatter_dropdown.bind(on_select=self.change_formatter)
        self.formatter_button.bind(on_release=self.formatter_dropdown.open)
        self.value_input.bind(on_text_validate=self.change_value)

    def change_value(self, instance: TextInput):
        if instance.text and self.selected:
            self.update_data(value=float(instance.text))

    def change_formatter(self, instance, value):
        self.update_data(formatter=value)

    @property
    def recycleview(self) -> RecycleView:
        return self.parent.parent

    @property
    def data(self) -> Data:
        return {
            "offset": self.offset,
            "value": self.value,
            "formatter": self.formatter
        }

    def update_data(self, dispatch: bool = True, **attrs: Data):
        attrs.update(
            previous_data=self.data,
            modified_attrs=list(attrs.keys())
        )
        if dispatch:
            self.recycleview.data[self.index] = {
                **self.recycleview.data[self.index],
                **attrs
            }
        else:
            self.recycleview.data[self.index].update(attrs)

    def refresh_view_attrs(self, rv: RecycleView, index: int, data: Data):
        self.index = index
        return super(RowData, self).refresh_view_attrs(
            rv, index, data)

    def on_touch_down(self, touch):
        if super(RowData, self).on_touch_down(touch):
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
        #Clock.schedule_interval(self.update_values, 1)

    def generate_data(self, count=5):
        for i in range(count):
            yield dict(offset=i, value=DEFAULT_VALUE, formatter='uint16')

    def change_data(self, model, **kwargs):
        if kwargs.get("modified"):
            for data in model.data[kwargs.get("modified")]:
                pass
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
