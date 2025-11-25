// Auto-generated. Do not edit!

// (in-package navigation_runner.srv)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;
let geometry_msgs = _finder('geometry_msgs');

//-----------------------------------------------------------


//-----------------------------------------------------------

class GetSafeActionMapRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.agent_position = null;
      this.agent_velocity = null;
      this.agent_size = null;
      this.dynamic_obs_position = null;
      this.dynamic_obs_velocity = null;
      this.dynamic_obs_size = null;
      this.static_obs_position = null;
      this.static_obs_size = null;
      this.static_obs_angle = null;
      this.max_velocity = null;
      this.rl_velocity = null;
    }
    else {
      if (initObj.hasOwnProperty('agent_position')) {
        this.agent_position = initObj.agent_position
      }
      else {
        this.agent_position = new geometry_msgs.msg.Point();
      }
      if (initObj.hasOwnProperty('agent_velocity')) {
        this.agent_velocity = initObj.agent_velocity
      }
      else {
        this.agent_velocity = new geometry_msgs.msg.Vector3();
      }
      if (initObj.hasOwnProperty('agent_size')) {
        this.agent_size = initObj.agent_size
      }
      else {
        this.agent_size = 0.0;
      }
      if (initObj.hasOwnProperty('dynamic_obs_position')) {
        this.dynamic_obs_position = initObj.dynamic_obs_position
      }
      else {
        this.dynamic_obs_position = [];
      }
      if (initObj.hasOwnProperty('dynamic_obs_velocity')) {
        this.dynamic_obs_velocity = initObj.dynamic_obs_velocity
      }
      else {
        this.dynamic_obs_velocity = [];
      }
      if (initObj.hasOwnProperty('dynamic_obs_size')) {
        this.dynamic_obs_size = initObj.dynamic_obs_size
      }
      else {
        this.dynamic_obs_size = [];
      }
      if (initObj.hasOwnProperty('static_obs_position')) {
        this.static_obs_position = initObj.static_obs_position
      }
      else {
        this.static_obs_position = [];
      }
      if (initObj.hasOwnProperty('static_obs_size')) {
        this.static_obs_size = initObj.static_obs_size
      }
      else {
        this.static_obs_size = [];
      }
      if (initObj.hasOwnProperty('static_obs_angle')) {
        this.static_obs_angle = initObj.static_obs_angle
      }
      else {
        this.static_obs_angle = [];
      }
      if (initObj.hasOwnProperty('max_velocity')) {
        this.max_velocity = initObj.max_velocity
      }
      else {
        this.max_velocity = 0.0;
      }
      if (initObj.hasOwnProperty('rl_velocity')) {
        this.rl_velocity = initObj.rl_velocity
      }
      else {
        this.rl_velocity = new geometry_msgs.msg.Vector3();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetSafeActionMapRequest
    // Serialize message field [agent_position]
    bufferOffset = geometry_msgs.msg.Point.serialize(obj.agent_position, buffer, bufferOffset);
    // Serialize message field [agent_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.agent_velocity, buffer, bufferOffset);
    // Serialize message field [agent_size]
    bufferOffset = _serializer.float64(obj.agent_size, buffer, bufferOffset);
    // Serialize message field [dynamic_obs_position]
    // Serialize the length for message field [dynamic_obs_position]
    bufferOffset = _serializer.uint32(obj.dynamic_obs_position.length, buffer, bufferOffset);
    obj.dynamic_obs_position.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [dynamic_obs_velocity]
    // Serialize the length for message field [dynamic_obs_velocity]
    bufferOffset = _serializer.uint32(obj.dynamic_obs_velocity.length, buffer, bufferOffset);
    obj.dynamic_obs_velocity.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [dynamic_obs_size]
    // Serialize the length for message field [dynamic_obs_size]
    bufferOffset = _serializer.uint32(obj.dynamic_obs_size.length, buffer, bufferOffset);
    obj.dynamic_obs_size.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [static_obs_position]
    // Serialize the length for message field [static_obs_position]
    bufferOffset = _serializer.uint32(obj.static_obs_position.length, buffer, bufferOffset);
    obj.static_obs_position.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [static_obs_size]
    // Serialize the length for message field [static_obs_size]
    bufferOffset = _serializer.uint32(obj.static_obs_size.length, buffer, bufferOffset);
    obj.static_obs_size.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [static_obs_angle]
    bufferOffset = _arraySerializer.float64(obj.static_obs_angle, buffer, bufferOffset, null);
    // Serialize message field [max_velocity]
    bufferOffset = _serializer.float64(obj.max_velocity, buffer, bufferOffset);
    // Serialize message field [rl_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rl_velocity, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSafeActionMapRequest
    let len;
    let data = new GetSafeActionMapRequest(null);
    // Deserialize message field [agent_position]
    data.agent_position = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset);
    // Deserialize message field [agent_velocity]
    data.agent_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [agent_size]
    data.agent_size = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [dynamic_obs_position]
    // Deserialize array length for message field [dynamic_obs_position]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.dynamic_obs_position = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.dynamic_obs_position[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [dynamic_obs_velocity]
    // Deserialize array length for message field [dynamic_obs_velocity]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.dynamic_obs_velocity = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.dynamic_obs_velocity[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [dynamic_obs_size]
    // Deserialize array length for message field [dynamic_obs_size]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.dynamic_obs_size = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.dynamic_obs_size[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [static_obs_position]
    // Deserialize array length for message field [static_obs_position]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.static_obs_position = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.static_obs_position[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [static_obs_size]
    // Deserialize array length for message field [static_obs_size]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.static_obs_size = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.static_obs_size[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [static_obs_angle]
    data.static_obs_angle = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [max_velocity]
    data.max_velocity = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [rl_velocity]
    data.rl_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 24 * object.dynamic_obs_position.length;
    length += 24 * object.dynamic_obs_velocity.length;
    length += 24 * object.dynamic_obs_size.length;
    length += 24 * object.static_obs_position.length;
    length += 24 * object.static_obs_size.length;
    length += 8 * object.static_obs_angle.length;
    return length + 112;
  }

  static datatype() {
    // Returns string type for a service object
    return 'navigation_runner/GetSafeActionMapRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '867ddf34cca17e53dbb05daa27a78640';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Point agent_position
    geometry_msgs/Vector3 agent_velocity
    float64 agent_size
    
    # dynamic obstacles
    geometry_msgs/Vector3[] dynamic_obs_position
    geometry_msgs/Vector3[] dynamic_obs_velocity
    geometry_msgs/Vector3[] dynamic_obs_size
    
    # static obstacles
    geometry_msgs/Vector3[] static_obs_position
    geometry_msgs/Vector3[] static_obs_size
    float64[] static_obs_angle
    
    float64 max_velocity
    geometry_msgs/Vector3 rl_velocity
    
    ================================================================================
    MSG: geometry_msgs/Point
    # This contains the position of a point in free space
    float64 x
    float64 y
    float64 z
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetSafeActionMapRequest(null);
    if (msg.agent_position !== undefined) {
      resolved.agent_position = geometry_msgs.msg.Point.Resolve(msg.agent_position)
    }
    else {
      resolved.agent_position = new geometry_msgs.msg.Point()
    }

    if (msg.agent_velocity !== undefined) {
      resolved.agent_velocity = geometry_msgs.msg.Vector3.Resolve(msg.agent_velocity)
    }
    else {
      resolved.agent_velocity = new geometry_msgs.msg.Vector3()
    }

    if (msg.agent_size !== undefined) {
      resolved.agent_size = msg.agent_size;
    }
    else {
      resolved.agent_size = 0.0
    }

    if (msg.dynamic_obs_position !== undefined) {
      resolved.dynamic_obs_position = new Array(msg.dynamic_obs_position.length);
      for (let i = 0; i < resolved.dynamic_obs_position.length; ++i) {
        resolved.dynamic_obs_position[i] = geometry_msgs.msg.Vector3.Resolve(msg.dynamic_obs_position[i]);
      }
    }
    else {
      resolved.dynamic_obs_position = []
    }

    if (msg.dynamic_obs_velocity !== undefined) {
      resolved.dynamic_obs_velocity = new Array(msg.dynamic_obs_velocity.length);
      for (let i = 0; i < resolved.dynamic_obs_velocity.length; ++i) {
        resolved.dynamic_obs_velocity[i] = geometry_msgs.msg.Vector3.Resolve(msg.dynamic_obs_velocity[i]);
      }
    }
    else {
      resolved.dynamic_obs_velocity = []
    }

    if (msg.dynamic_obs_size !== undefined) {
      resolved.dynamic_obs_size = new Array(msg.dynamic_obs_size.length);
      for (let i = 0; i < resolved.dynamic_obs_size.length; ++i) {
        resolved.dynamic_obs_size[i] = geometry_msgs.msg.Vector3.Resolve(msg.dynamic_obs_size[i]);
      }
    }
    else {
      resolved.dynamic_obs_size = []
    }

    if (msg.static_obs_position !== undefined) {
      resolved.static_obs_position = new Array(msg.static_obs_position.length);
      for (let i = 0; i < resolved.static_obs_position.length; ++i) {
        resolved.static_obs_position[i] = geometry_msgs.msg.Vector3.Resolve(msg.static_obs_position[i]);
      }
    }
    else {
      resolved.static_obs_position = []
    }

    if (msg.static_obs_size !== undefined) {
      resolved.static_obs_size = new Array(msg.static_obs_size.length);
      for (let i = 0; i < resolved.static_obs_size.length; ++i) {
        resolved.static_obs_size[i] = geometry_msgs.msg.Vector3.Resolve(msg.static_obs_size[i]);
      }
    }
    else {
      resolved.static_obs_size = []
    }

    if (msg.static_obs_angle !== undefined) {
      resolved.static_obs_angle = msg.static_obs_angle;
    }
    else {
      resolved.static_obs_angle = []
    }

    if (msg.max_velocity !== undefined) {
      resolved.max_velocity = msg.max_velocity;
    }
    else {
      resolved.max_velocity = 0.0
    }

    if (msg.rl_velocity !== undefined) {
      resolved.rl_velocity = geometry_msgs.msg.Vector3.Resolve(msg.rl_velocity)
    }
    else {
      resolved.rl_velocity = new geometry_msgs.msg.Vector3()
    }

    return resolved;
    }
};

class GetSafeActionMapResponse {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.safe_action = null;
    }
    else {
      if (initObj.hasOwnProperty('safe_action')) {
        this.safe_action = initObj.safe_action
      }
      else {
        this.safe_action = new geometry_msgs.msg.Vector3();
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type GetSafeActionMapResponse
    // Serialize message field [safe_action]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.safe_action, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSafeActionMapResponse
    let len;
    let data = new GetSafeActionMapResponse(null);
    // Deserialize message field [safe_action]
    data.safe_action = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'navigation_runner/GetSafeActionMapResponse';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '4b9504256e99d21a75b4954b17574a23';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Vector3 safe_action
    
    ================================================================================
    MSG: geometry_msgs/Vector3
    # This represents a vector in free space. 
    # It is only meant to represent a direction. Therefore, it does not
    # make sense to apply a translation to it (e.g., when applying a 
    # generic rigid transformation to a Vector3, tf2 will only apply the
    # rotation). If you want your data to be translatable too, use the
    # geometry_msgs/Point message instead.
    
    float64 x
    float64 y
    float64 z
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new GetSafeActionMapResponse(null);
    if (msg.safe_action !== undefined) {
      resolved.safe_action = geometry_msgs.msg.Vector3.Resolve(msg.safe_action)
    }
    else {
      resolved.safe_action = new geometry_msgs.msg.Vector3()
    }

    return resolved;
    }
};

module.exports = {
  Request: GetSafeActionMapRequest,
  Response: GetSafeActionMapResponse,
  md5sum() { return '4635b39c982fd59eaaa757b70b204f4c'; },
  datatype() { return 'navigation_runner/GetSafeActionMap'; }
};
