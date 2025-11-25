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

class GetSafeActionRequest {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.agent_position = null;
      this.agent_velocity = null;
      this.agent_size = null;
      this.obs_position = null;
      this.obs_velocity = null;
      this.obs_size = null;
      this.laser_points = null;
      this.laser_range = null;
      this.laser_res = null;
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
      if (initObj.hasOwnProperty('obs_position')) {
        this.obs_position = initObj.obs_position
      }
      else {
        this.obs_position = [];
      }
      if (initObj.hasOwnProperty('obs_velocity')) {
        this.obs_velocity = initObj.obs_velocity
      }
      else {
        this.obs_velocity = [];
      }
      if (initObj.hasOwnProperty('obs_size')) {
        this.obs_size = initObj.obs_size
      }
      else {
        this.obs_size = [];
      }
      if (initObj.hasOwnProperty('laser_points')) {
        this.laser_points = initObj.laser_points
      }
      else {
        this.laser_points = [];
      }
      if (initObj.hasOwnProperty('laser_range')) {
        this.laser_range = initObj.laser_range
      }
      else {
        this.laser_range = 0.0;
      }
      if (initObj.hasOwnProperty('laser_res')) {
        this.laser_res = initObj.laser_res
      }
      else {
        this.laser_res = 0.0;
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
    // Serializes a message object of type GetSafeActionRequest
    // Serialize message field [agent_position]
    bufferOffset = geometry_msgs.msg.Point.serialize(obj.agent_position, buffer, bufferOffset);
    // Serialize message field [agent_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.agent_velocity, buffer, bufferOffset);
    // Serialize message field [agent_size]
    bufferOffset = _serializer.float64(obj.agent_size, buffer, bufferOffset);
    // Serialize message field [obs_position]
    // Serialize the length for message field [obs_position]
    bufferOffset = _serializer.uint32(obj.obs_position.length, buffer, bufferOffset);
    obj.obs_position.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [obs_velocity]
    // Serialize the length for message field [obs_velocity]
    bufferOffset = _serializer.uint32(obj.obs_velocity.length, buffer, bufferOffset);
    obj.obs_velocity.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [obs_size]
    // Serialize the length for message field [obs_size]
    bufferOffset = _serializer.uint32(obj.obs_size.length, buffer, bufferOffset);
    obj.obs_size.forEach((val) => {
      bufferOffset = geometry_msgs.msg.Vector3.serialize(val, buffer, bufferOffset);
    });
    // Serialize message field [laser_points]
    bufferOffset = _arraySerializer.float64(obj.laser_points, buffer, bufferOffset, null);
    // Serialize message field [laser_range]
    bufferOffset = _serializer.float64(obj.laser_range, buffer, bufferOffset);
    // Serialize message field [laser_res]
    bufferOffset = _serializer.float64(obj.laser_res, buffer, bufferOffset);
    // Serialize message field [max_velocity]
    bufferOffset = _serializer.float64(obj.max_velocity, buffer, bufferOffset);
    // Serialize message field [rl_velocity]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.rl_velocity, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSafeActionRequest
    let len;
    let data = new GetSafeActionRequest(null);
    // Deserialize message field [agent_position]
    data.agent_position = geometry_msgs.msg.Point.deserialize(buffer, bufferOffset);
    // Deserialize message field [agent_velocity]
    data.agent_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    // Deserialize message field [agent_size]
    data.agent_size = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [obs_position]
    // Deserialize array length for message field [obs_position]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.obs_position = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.obs_position[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [obs_velocity]
    // Deserialize array length for message field [obs_velocity]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.obs_velocity = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.obs_velocity[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [obs_size]
    // Deserialize array length for message field [obs_size]
    len = _deserializer.uint32(buffer, bufferOffset);
    data.obs_size = new Array(len);
    for (let i = 0; i < len; ++i) {
      data.obs_size[i] = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset)
    }
    // Deserialize message field [laser_points]
    data.laser_points = _arrayDeserializer.float64(buffer, bufferOffset, null)
    // Deserialize message field [laser_range]
    data.laser_range = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [laser_res]
    data.laser_res = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [max_velocity]
    data.max_velocity = _deserializer.float64(buffer, bufferOffset);
    // Deserialize message field [rl_velocity]
    data.rl_velocity = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    let length = 0;
    length += 24 * object.obs_position.length;
    length += 24 * object.obs_velocity.length;
    length += 24 * object.obs_size.length;
    length += 8 * object.laser_points.length;
    return length + 120;
  }

  static datatype() {
    // Returns string type for a service object
    return 'navigation_runner/GetSafeActionRequest';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return 'cf1c70d551fddbbb7abfbebc08b9923f';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    geometry_msgs/Point agent_position
    geometry_msgs/Vector3 agent_velocity
    float64 agent_size
    
    # dynamic obstacles
    geometry_msgs/Vector3[] obs_position
    geometry_msgs/Vector3[] obs_velocity
    geometry_msgs/Vector3[] obs_size
    
    # static obstacles
    float64[] laser_points
    float64 laser_range
    float64 laser_res # use the larger one among horizonal and vertical
    
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
    const resolved = new GetSafeActionRequest(null);
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

    if (msg.obs_position !== undefined) {
      resolved.obs_position = new Array(msg.obs_position.length);
      for (let i = 0; i < resolved.obs_position.length; ++i) {
        resolved.obs_position[i] = geometry_msgs.msg.Vector3.Resolve(msg.obs_position[i]);
      }
    }
    else {
      resolved.obs_position = []
    }

    if (msg.obs_velocity !== undefined) {
      resolved.obs_velocity = new Array(msg.obs_velocity.length);
      for (let i = 0; i < resolved.obs_velocity.length; ++i) {
        resolved.obs_velocity[i] = geometry_msgs.msg.Vector3.Resolve(msg.obs_velocity[i]);
      }
    }
    else {
      resolved.obs_velocity = []
    }

    if (msg.obs_size !== undefined) {
      resolved.obs_size = new Array(msg.obs_size.length);
      for (let i = 0; i < resolved.obs_size.length; ++i) {
        resolved.obs_size[i] = geometry_msgs.msg.Vector3.Resolve(msg.obs_size[i]);
      }
    }
    else {
      resolved.obs_size = []
    }

    if (msg.laser_points !== undefined) {
      resolved.laser_points = msg.laser_points;
    }
    else {
      resolved.laser_points = []
    }

    if (msg.laser_range !== undefined) {
      resolved.laser_range = msg.laser_range;
    }
    else {
      resolved.laser_range = 0.0
    }

    if (msg.laser_res !== undefined) {
      resolved.laser_res = msg.laser_res;
    }
    else {
      resolved.laser_res = 0.0
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

class GetSafeActionResponse {
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
    // Serializes a message object of type GetSafeActionResponse
    // Serialize message field [safe_action]
    bufferOffset = geometry_msgs.msg.Vector3.serialize(obj.safe_action, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type GetSafeActionResponse
    let len;
    let data = new GetSafeActionResponse(null);
    // Deserialize message field [safe_action]
    data.safe_action = geometry_msgs.msg.Vector3.deserialize(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 24;
  }

  static datatype() {
    // Returns string type for a service object
    return 'navigation_runner/GetSafeActionResponse';
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
    const resolved = new GetSafeActionResponse(null);
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
  Request: GetSafeActionRequest,
  Response: GetSafeActionResponse,
  md5sum() { return '54fa8c288cd542a68cd6fb86a87e59f8'; },
  datatype() { return 'navigation_runner/GetSafeAction'; }
};
