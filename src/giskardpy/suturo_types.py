from enum import unique, Enum


# States for the HSRs gripper, is being used only in old_python_interface as of now
@unique
class GripperTypes(Enum):
    OPEN = 'open'
    CLOSE = 'close'
    NEUTRAL = 'neutral'


# List of Methods that need Thresholds for force_monitor: GraspObjectCarefully(might include Doors?), Placing
@unique
class ForceTorqueThresholds(Enum):
    FT_GraspWithCare = 'GraspCarefully'
    FT_PlaceCutlery = 'PlaceCutlery'
    FT_Placing = 'Place'
    FT_Door = 'Door'
    FT_DishDoor = 'DishDoor'
    FT_Tilt = 'Tilt'  # Pouring
    # FT_Shelf_Grasp = 'FT_Shelf' # Might not be needed at all


# List of Objects that need to be differentiated between when placing method is used
@unique
class ObjectTypes(Enum):
    OT_Standard = 'Standard'  # Normal Objects(e.g Milk), includes Cups/Glasses, since planning grabs them from front
    OT_Cutlery = 'Cutlery'
    OT_Plate = 'Plate'
    OT_Bowl = 'Bowl'


# List of possible grasping directions
@unique
class GraspTypes(Enum):
    FRONT = 'front'
    ABOVE = 'above'
    LEFT = 'left'
    RIGHT = 'right'
    BELOW = 'below'
