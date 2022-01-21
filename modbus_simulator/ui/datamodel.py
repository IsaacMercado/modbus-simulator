from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from random import uniform, randint
from copy import deepcopy

from kivy.lang import Builder
from kivy.logger import Logger
from kivy.properties import (
    BooleanProperty,
    ObjectProperty,
    NumericProperty,
    StringProperty
)

from kivy.uix.recycleview import RecycleView
from kivy.uix.recycleview.views import RecycleKVIDsDataViewBehavior
from kivy.uix.label import Label
from kivy.uix.recycleboxlayout import RecycleBoxLayout
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.recycleview.layout import LayoutSelectionBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.dropdown import DropDown
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup

from ..utils.background_job import BackgroundJob
from .conts import TEMPLATES_DIR, DEFAULT_VALUE, DEFAULT_FORMATTER, TYPE_RANGES, Data, Number

__all__ = [
    "ErrorPopup", "TypeDropDown", "SelectableRecycleBoxLayout", "RowData", "DataModel"
]

Builder.load_file(str(TEMPLATES_DIR.joinpath('datamodel.kv')))


class ErrorPopup(Popup):
    """
    Popup class to display error messages
    """

    def __init__(self, title, text, **kwargs):
        super(ErrorPopup, self).__init__()
        # super(ErrorPopup, self).__init__(**kwargs)

        content = BoxLayout(orientation="vertical")
        content.add_widget(Label(text=text, font_size=20))

        button = Button(text="Dismiss", size_hint=(1, .20), font_size=20)
        button.bind(on_release=lambda *args: self.dismiss())
        content.add_widget(button)

        self.content = content
        self.title = title
        self.auto_dismiss = False
        self.size_hint = .7, .5
        self.font_size = 20

        self.open()


class TypeDropDown(DropDown):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        _, *dtypes = TYPE_RANGES.keys()

        for dtype in dtypes:
            button = Button(text=dtype, size_hint_y=None, height=45,
                            background_color=[0.0, 0.5, 1.0, 1.0])
            button.bind(on_release=partial(self.select_type, dtype))
            self.add_widget(button)

    def select_type(self, dtype: str, instance: DropDown):
        self.select(dtype)


class SelectableRecycleBoxLayout(FocusBehavior,
                                 LayoutSelectionBehavior,
                                 RecycleBoxLayout):
    pass


class RowData(RecycleKVIDsDataViewBehavior, BoxLayout):
    index = None
    selected = BooleanProperty(False)
    selectable = BooleanProperty(True)

    offset = NumericProperty()
    value = NumericProperty(DEFAULT_VALUE)
    formatter = StringProperty(DEFAULT_FORMATTER)

    offset_label: Label = ObjectProperty()
    value_input: TextInput = ObjectProperty()
    formatter_button: Button = ObjectProperty()
    formatter_dropdown: Optional[DropDown] = ObjectProperty()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.formatter_dropdown = TypeDropDown()
        self.formatter_dropdown.bind(
            on_select=lambda instance, value: self.update_data(formatter=value)
        )
        self.formatter_button.bind(on_release=self.formatter_dropdown.open)
        self.value_input.bind(on_text_validate=self.change_value)

    @property
    def recycleview(self) -> RecycleView:
        return self.parent.recycleview

    def change_value(self, instance: TextInput):
        minval, maxval = self.recycleview.minval, self.recycleview.maxval
        is_continuous = self.recycleview.is_continuous
        text = instance.text

        if text:
            value = float(text) if is_continuous else int(text)

            if value >= minval and value <= maxval:
                self.update_data(value=value, disabled=True)
                # self.deselect()
                return

        error_text = f"Only numeric value in range {minval}-{maxval} to be used"
        ErrorPopup(title="Error", text=error_text)
        self.update_data(**{"value_input.hint_text": error_text})

    @property
    def data(self) -> Data:
        return {
            "offset": self.offset,
            "value": self.value,
            "formatter": self.formatter
        }

    def update_data(self, dispatch: bool = True, **attrs: Data) -> None:
        attrs.update(
            previous_data=self.data,
            modified_attrs=set(attrs.keys())
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
            return True
        if self.collide_point(*touch.pos) and self.selectable:
            return self.parent.select_with_touch(self.index, touch)

    def apply_selection(self, rv, index, is_selected):
        rv.data[index]['selected'] = self.selected = is_selected


class DataModel(RecycleView):

    minval: Number = NumericProperty(0)
    maxval: Number = NumericProperty(1)
    _parent: Any = ObjectProperty()

    simulate = False
    time_interval = 1
    dirty_thread = False
    dirty_model = False
    simulate_timer = None
    simulate = False
    dispatcher = None
    list_view = None
    is_simulating = False
    blockname = "<BLOCK_NAME_NOT_SET>"

    def __init__(self, **kwargs):
        super(DataModel, self).__init__(**kwargs)
        self.data_model.bind(on_data_changed=self.change_data)
        self.init()

    def init(
        self,
        simulate: bool = False,
        time_interval: Number = 1,
        minval: Optional[Number] = None,
        maxval: Optional[Number] = None,
        blockname: Optional[Number] = None,
        parent: Any = None,
    ):
        """Initializes Datamodel"""

        self.minval = minval or self.minval
        self.maxval = maxval or self.maxval
        self.blockname = blockname or self.blockname
        self.simulate = simulate
        self.time_interval = time_interval
        self.enable_type_selection = self.is_continuous

        self._parent = parent or self._parent
        self.simulate_timer = BackgroundJob(
            "simulation",
            self.time_interval,
            self._simulate_block_values
        )

    def reinit(
        self,
        time_interval: Optional[Number] = None,
        minval: Optional[Number] = None,
        maxval: Optional[Number] = None
    ):
        """
        Re-initializes Datamodel on change in model configuration from settings
        """
        self.minval = minval or self.minval
        self.maxval = maxval or self.maxval
        try:
            if time_interval and int(time_interval) != self.time_interval:
                self.time_interval = time_interval
                if self.is_simulating:
                    self.simulate_timer.cancel()
                self.simulate_timer = BackgroundJob(
                    "simulation",
                    self.time_interval,
                    self._simulate_block_values
                )
                self.dirty_thread = False
                self.start_stop_simulation(self.simulate)
        except ValueError:
            kwargs = dict(
                minval=minval,
                maxval=maxval,
                time_interval=time_interval
            )
            Logger.debug("Error while reinitializing DataModel %s" % kwargs)

    @property
    def is_continuous(self) -> bool:
        return self.blockname in {'input_registers', 'holding_registers'}

    @property
    def is_run(self) -> bool:
        return self.simulate and self.data

    @property
    def selection(self) -> Tuple[Data]:
        return tuple(
            filter(lambda item: item.get('selected'), self.data)
        )

    @property
    def offsets(self) -> Tuple[Union[int, None]]:
        return tuple(
            map(lambda data: data.get("offset"), self.data)
        )

    def get_address(self, offset) -> int:
        if self.blockname == "coils":
            offset = offset
        elif self.blockname == "discrete_inputs":
            offset = 10001 + offset if offset < 10001 else offset
        elif self.blockname == "input_registers":
            offset = 30001 + offset if offset < 30001 else offset
        else:
            offset = 40001 + offset if offset < 40001 else offset

        return offset

    def add_data(self, data: Dict):
        """
        Adds data to the Data model
        :param data:
        :return:
        """
        item_strings = []
        next_index = max(self.offsets) + 1 if self.offsets else 0

        data = [{
            'offset': self.get_address(int(offset) + next_index),
            **data
        } for offset, data in data.items()]

        for index in range(len(data)):
            offset = data[index]['offset']
            item_strings.append(offset)

            if not data[index].get('formatter'):
                if int(offset) >= 30001:
                    data[index]['formatter'] = 'uint16'
                else:
                    data[index]['formatter'] = 'boolean'

        self.data.extend(data)

        return self.data, tuple(map(str, item_strings))

    def delete_data(self, item_strings: List[str]):
        """
        Delete data from data model
        :param item_strings:
        :return:
        """
        items_popped = []

        for item in self.selection:
            index_popped = item_strings.pop(item_strings.index(int(item.text)))
            self.list_view.adapter.data.pop(int(item.text), None)
            self.list_view.adapter.update_for_new_data()
            self.list_view._trigger_reset_populate()
            items_popped.append(index_popped)

        return items_popped, self.data

    def change_data(self, model, **kwargs):
        # print(kwargs)
        if "modified" in kwargs:
            for data in model.data[kwargs.get("modified")]:
                modified_attrs = data.get("modified_attrs", ())

                if not modified_attrs:
                    continue

                base_data = self.extract_data(data)

                Logger.debug(f"Update {self.blockname} data: {base_data}")

                if "value" in modified_attrs:
                    if self._parent:
                        self._parent.sync_data_callback(
                            self.blockname,
                            base_data
                        )

                    if data.get("selected", False):
                        try:
                            index = self.offsets.index(data.get("offset"))
                            self.data[index] = self.update_data(
                                self.data[index],
                                disabled=False
                            )
                        except ValueError:
                            pass

                elif "formatter" in modified_attrs:
                    if self._parent:
                        self._parent.sync_formatter_callback(
                            self.blockname,
                            base_data,
                            data.get('previous_data', {}).get('formatter')
                        )

    def set_data(self, data: Data, **changes: Data):
        index = None

        try:
            index = self.data.index(data)
        except ValueError:
            try:
                index = self.offsets.index(data.get("offset"))
            except ValueError:
                pass

        if index:
            self.data[index] = self.update_data(data, **changes)

    def extract_data(self, data: Data):
        return {
            key: data.get(key) for key in ("offset", "value", "formatter")
        }

    def update_data(self, data: Data, **changes: Data) -> Data:
        data = data.copy()
        data.update(
            previous_data=self.extract_data(data),
            modified_attrs=set(changes.keys())
        )
        data.update(changes)
        return data

    def start_stop_simulation(self, simulate):
        """
        Starts or stops simulating data
        :param simulate:
        :return:
        """
        self.simulate = simulate

        if self.simulate:
            if self.dirty_thread:
                self.simulate_timer = BackgroundJob(
                    "simulation",
                    self.time_interval,
                    self._simulate_block_values
                )
            self.simulate_timer.start()
            self.dirty_thread = False
            self.is_simulating = True
        else:
            self.simulate_timer.cancel()
            self.dirty_thread = True
            self.is_simulating = False

    def random_value_from_data(self, data: Data) -> Data:
        formatter = data.get('formatter')
        minval, maxval = TYPE_RANGES.get(formatter, (0, 1))
        minval = minval if minval > self.minval else self.minval
        maxval = maxval if maxval < self.maxval else self.maxval

        if formatter.startswith('float'):
            value = round(uniform(minval, maxval), 2)
        else:
            value = randint(minval, maxval)

        return self.update_data(data, value=value)

    def _simulate_block_values(self, *args) -> None:
        if self.is_run:
            self.data[0:len(self.data)] = map(
                self.random_value_from_data,
                self.data
            )

    def reset_block_values(self) -> None:
        if not self.is_run:
            self.data[0:len(self.data)] = map(
                partial(self.update_data, value=DEFAULT_VALUE),
                self.data
            )
