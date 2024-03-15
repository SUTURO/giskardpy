from enum import unique, Enum


# States for the HSRs gripper, is being used only in old_python_interface as of now
@unique
class GripperTypes(Enum):
    OPEN = 'open'
    CLOSE = 'close'
    NEUTRAL = 'neutral'


# List of Methods that need Thresholds for force_monitor: GraspObjectCarefully(might include Doors?), Placing,
# TODO: Create Enums for every Force and/or Torque sensitive action, to be used in force_torque_monitor.py
@unique
class ForceTorqueThresholds(Enum):
    FT_GraspWithCare = 'FTGraspC'  # For any "Conventional Object" (for other objects use extra enum?)
    FT_Placing = 'FTPlace'
    FT_Door = 'FTDoor'
    FT_DishDoor = 'FTDishDoor'
    FT_GraspCutlery = 'FTGraspCutlery'
    FT_Tilt = 'FTTilt'  # Pouring
    # FT_Shelf_Grasp = 'FT_Shelf'
