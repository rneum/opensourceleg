import glob

import numpy as np

from opensourceleg.actuators.base import CONTROL_MODES
from opensourceleg.actuators.dephy import DephyActuator
from opensourceleg.control.fsm import State, StateMachine
from opensourceleg.logging.logger import Logger
from opensourceleg.robots.osl import OpenSourceLeg
from opensourceleg.sensors.encoder import AS5048B
from opensourceleg.sensors.loadcell import DephyLoadcellAmplifier
from opensourceleg.utilities import SoftRealtimeLoop

GEAR_RATIO = 9 * (83 / 18)
FREQUENCY = 200
LOADCELL_CALIBRATION_MATRIX = np.array([
    [5.14165, -1369.65674, 22.45444, 27.38696, -11.19945, 1346.30042],
    [-34.78300, 789.09161, -34.96016, -1533.74902, 23.92590, 778.18195],
    [-835.46417, 4.72628, -828.73022, -8.84200, -829.07013, 5.80179],
    [16.55477, 0.23425, -0.59845, 0.11180, -17.13909, 0.02438],
    [-10.19001, -0.43203, 19.44527, 0.01228, 9.65720, 0.48793],
    [-0.48663, -21.04720, -0.07773, -19.87712, 0.10256, -21.40916],
])


# ------------- TUNABLE FSM PARAMETERS ---------------- #
BODY_WEIGHT = 10 * 9.8  # 30 * 9.8

# STATE 1: EARLY STANCE
KNEE_K_ESTANCE = 99.372
KNEE_B_ESTANCE = 3.180
KNEE_THETA_ESTANCE = 5
ANKLE_K_ESTANCE = 19.874
ANKLE_B_ESTANCE = 0
ANKLE_THETA_ESTANCE = -2
LOAD_LSTANCE: float = 1.0 * BODY_WEIGHT * 0.25
ANKLE_THETA_ESTANCE_TO_LSTANCE = np.deg2rad(6.0)

# STATE 2: LATE STANCE
KNEE_K_LSTANCE = 99.372
KNEE_B_LSTANCE = 1.272
KNEE_THETA_LSTANCE = 8
ANKLE_K_LSTANCE = 79.498
ANKLE_B_LSTANCE = 0.063
ANKLE_THETA_LSTANCE = -20
LOAD_ESWING: float = 1.0 * BODY_WEIGHT * 0.15

# STATE 3: EARLY SWING
KNEE_K_ESWING = 39.749
KNEE_B_ESWING = 0.063
KNEE_THETA_ESWING = 60
ANKLE_K_ESWING = 7.949
ANKLE_B_ESWING = 0.0
ANKLE_THETA_ESWING = 25
KNEE_THETA_ESWING_TO_LSWING = np.deg2rad(50)
KNEE_DTHETA_ESWING_TO_LSWING = 3

# STATE 4: LATE SWING
KNEE_K_LSWING = 15.899
KNEE_B_LSWING = 3.816
KNEE_THETA_LSWING = 5
ANKLE_K_LSWING = 7.949
ANKLE_B_LSWING = 0.0
ANKLE_THETA_LSWING = 15
LOAD_ESTANCE: float = 1.0 * BODY_WEIGHT * 0.4
KNEE_THETA_LSWING_TO_ESTANCE = np.deg2rad(30)

# ---------------------------------------------------- #


def create_simple_walking_fsm(osl: OpenSourceLeg) -> StateMachine:
    e_stance = State(
        name="e_stance",
        knee_theta=KNEE_THETA_ESTANCE,
        knee_stiffness=KNEE_K_ESTANCE,
        knee_damping=KNEE_B_ESTANCE,
        ankle_theta=ANKLE_THETA_ESTANCE,
        ankle_stiffness=ANKLE_K_ESTANCE,
        ankle_damping=ANKLE_B_ESTANCE,
    )

    l_stance = State(
        name="l_stance",
        knee_theta=KNEE_THETA_LSTANCE,
        knee_stiffness=KNEE_K_LSTANCE,
        knee_damping=KNEE_B_LSTANCE,
        ankle_theta=ANKLE_THETA_LSTANCE,
        ankle_stiffness=ANKLE_K_LSTANCE,
        ankle_damping=ANKLE_B_LSTANCE,
    )

    e_swing = State(
        name="e_swing",
        knee_theta=KNEE_THETA_ESWING,
        knee_stiffness=KNEE_K_ESWING,
        knee_damping=KNEE_B_ESWING,
        ankle_theta=ANKLE_THETA_ESWING,
        ankle_stiffness=ANKLE_K_ESWING,
        ankle_damping=ANKLE_B_ESWING,
    )

    l_swing = State(
        name="l_swing",
        knee_theta=KNEE_THETA_LSWING,
        knee_stiffness=KNEE_K_LSWING,
        knee_damping=KNEE_B_LSWING,
        ankle_theta=ANKLE_THETA_LSWING,
        ankle_stiffness=ANKLE_K_LSWING,
        ankle_damping=ANKLE_B_LSWING,
    )

    def estance_to_lstance(osl: OpenSourceLeg) -> bool:
        """
        Transition from early stance to late stance when the loadcell
        reads a force greater than a threshold.
        """
        if osl.loadcell is None:
            raise ValueError("Loadcell is not connected")
        return bool(osl.loadcell.fz > LOAD_LSTANCE and osl.ankle.output_position > ANKLE_THETA_ESTANCE_TO_LSTANCE)

    def lstance_to_eswing(osl: OpenSourceLeg) -> bool:
        """
        Transition from late stance to early swing when the loadcell
        reads a force less than a threshold.
        """
        if osl.loadcell is None:
            raise ValueError("Loadcell is not connected")
        return bool(osl.loadcell.fz < LOAD_ESWING)

    def eswing_to_lswing(osl: OpenSourceLeg) -> bool:
        """
        Transition from early swing to late swing when the knee angle
        is greater than a threshold and the knee velocity is less than
        a threshold.
        """
        if osl.knee is None:
            raise ValueError("Knee is not connected")
        return bool(
            osl.knee.output_position > KNEE_THETA_ESWING_TO_LSWING
            and osl.knee.output_velocity < KNEE_DTHETA_ESWING_TO_LSWING
        )

    def lswing_to_estance(osl: OpenSourceLeg) -> bool:
        """
        Transition from late swing to early stance when the loadcell
        reads a force greater than a threshold or the knee angle is
        less than a threshold.
        """
        if osl.knee is None:
            raise ValueError("Knee is not connected")
        if osl.loadcell is None:
            raise ValueError("Loadcell is not connected")
        return bool(osl.loadcell.fz > LOAD_ESTANCE or osl.knee.output_position < KNEE_THETA_LSWING_TO_ESTANCE)

    fsm = StateMachine(
        states=[
            e_stance,
            l_stance,
            e_swing,
            l_swing,
        ],
        initial_state_name="e_stance",
    )

    fsm.add_transition(
        source=e_stance,
        destination=l_stance,
        event_name="foot_flat",
        criteria=estance_to_lstance,
    )
    fsm.add_transition(
        source=l_stance,
        destination=e_swing,
        event_name="heel_off",
        criteria=lstance_to_eswing,
    )
    fsm.add_transition(
        source=e_swing,
        destination=l_swing,
        event_name="toe_off",
        criteria=eswing_to_lswing,
    )
    fsm.add_transition(
        source=l_swing,
        destination=e_stance,
        event_name="heel_strike",
        criteria=lswing_to_estance,
    )
    return fsm


if __name__ == "__main__":
    # get list of all usb connections
    ports = glob.glob("/dev/ttyACM*")

    # for now, expect exactly two dephy actuators on ttyACM
    if len(ports) > 2:
        print("more than two devices detected")
    else:
        for i in ports:
            actuatorTemp = DephyActuator(port=i)
            actuatorTemp.start()
            if actuatorTemp.id == 1439:
                knee_port = i
            elif actuatorTemp.id == 1423:
                ankle_port = i
            else:
                print(actuatorTemp.id)
                print("I've never seen that actuator in my life")
            actuatorTemp.stop()

    actuators = {
        "knee": DephyActuator(
            tag="knee",
            port=knee_port,
            gear_ratio=GEAR_RATIO,
            frequency=FREQUENCY,
            debug_level=0,
            dephy_log=False,
        ),
        "ankle": DephyActuator(
            tag="ankle",
            port=ankle_port,
            gear_ratio=GEAR_RATIO,
            frequency=FREQUENCY,
            debug_level=0,
            dephy_log=False,
        ),
    }

    sensors = {
        "loadcell": DephyLoadcellAmplifier(
            calibration_matrix=LOADCELL_CALIBRATION_MATRIX,
        ),
        "joint_encoder_knee": AS5048B(
            tag="joint_encoder_knee",
            bus="/dev/i2c-1",
            A1_adr_pin=False,
            A2_adr_pin=True,
            zero_position=0,
            enable_diagnostics=False,
        ),
        "joint_encoder_ankle": AS5048B(
            tag="joint_encoder_ankle",
            bus="/dev/i2c-1",
            A1_adr_pin=False,
            A2_adr_pin=False,
            zero_position=0,
            enable_diagnostics=False,
        ),
    }

    clock = SoftRealtimeLoop(dt=1 / FREQUENCY)
    fsm_logger = Logger(
        log_path="./logs",
        file_name="fsm.log",
    )

    osl = OpenSourceLeg(
        tag="osl",
        actuators=actuators,
        sensors=sensors,
    )

    osl_fsm = create_simple_walking_fsm(osl)

    # Zeroing the joint encoders
    def knee_homing_complete():
        osl.joint_encoder_knee.update()
        osl.joint_encoder_knee.zero_position = osl.joint_encoder_knee.counts
        print("Knee homing complete!")

    def ankle_homing_complete():
        osl.joint_encoder_ankle.update()
        # The hard stop for ankle is at 30 deg from the zero position
        osl.joint_encoder_ankle.zero_position = osl.joint_encoder_ankle.counts - osl.joint_encoder_ankle.deg_to_counts(
            30
        )
        print("Ankle homing complete!")

    callbacks = {"knee": knee_homing_complete, "ankle": ankle_homing_complete}

    with osl, osl_fsm:
        osl.update()
        osl.home(callbacks=callbacks)
        input("Press Enter to start walking...")

        # knee
        osl.knee.set_control_mode(mode=CONTROL_MODES.IMPEDANCE)
        osl.knee.set_impedance_cc_pidf_gains()
        osl.knee.set_output_impedance()

        # ankle
        osl.ankle.set_control_mode(mode=CONTROL_MODES.IMPEDANCE)
        osl.ankle.set_impedance_cc_pidf_gains()
        osl.ankle.set_output_impedance()

        osl.loadcell.reset()
        osl.loadcell.calibrate()

        for t in clock:
            osl.update()
            print("Ankle position", np.rad2deg(osl.sensors["joint_encoder_ankle"].position))
            print("Knee position", np.rad2deg(osl.sensors["joint_encoder_knee"].position))
            osl_fsm.update(osl=osl)
            # osl.knee.set_output_impedance(
            #     k=osl_fsm.current_state.knee_stiffness,
            #     b=osl_fsm.current_state.knee_damping,
            # )
            # osl.ankle.set_output_impedance(
            #     k=osl_fsm.current_state.ankle_stiffness,
            #     b=osl_fsm.current_state.ankle_damping,
            # )

            # osl.knee.set_output_position(np.deg2rad(osl_fsm.current_state.knee_theta))
            # osl.ankle.set_output_position(np.deg2rad(osl_fsm.current_state.ankle_theta))

            fsm_logger.info(
                f"T: {t:.3f}s, "
                f"Current state: {osl_fsm.current_state.name}; "
                f"Loadcell Fz: {osl.loadcell.fz:.3f} N; "
                f"Knee theta: {np.rad2deg(osl.knee.output_position):.3f} deg; "
                f"Ankle theta: {np.rad2deg(osl.ankle.output_position):.3f} deg; "
                f"Knee winding temperature: {osl.knee.winding_temperature:.3f} c; "
                f"Ankle winding temperature: {osl.ankle.winding_temperature:.3f} c; "
            )
