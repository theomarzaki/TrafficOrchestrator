const net = require('net');


const PORT = 8124;

class ToTCPServer {

    constructor() {
        this.server = net.createServer((socket) => {
            socket.on('data', (data) => {
                this.handleData(data.toString('utf8'), socket);
            });

            socket.on('error', (err) => {
                console.log("socket error ", err.toString());
                // when error occur we need to stop trying to send data to this socket
                const subscriptionId = socket.subscriptionId;
                toMessageHandler.clearTimers(subscriptionId);
                socket.destroy();
            });

            socket.on('end', () => {
                console.log("client disconnected");
            });
        });

        this.server.on('listening', () => {
            console.log(`TO TCP Server listening on port ${PORT}`);
        });
        this.server.on('connection', () => {
            console.log('Client connected');
        });
        this.server.on('close', () => {
            console.log('TO TCP Server closed');
        });
        this.server.on('error', (err) => {
            console.log('TO TCP server throw error', err);
        });
    }

    start() {
        this.server.listen(PORT, '0.0.0.0');
    }

    handleData(data, socket) {
        try {
            const message = JSON.parse(data);
            console.log("message Type : " +message.type);
            switch (message.type) {
                case "subscription_request":
                	console.log("in case");
                    this.handleSubscriptionRequest(message, socket);
                    break;
                default:
                    console.log(`received unsupported message type ${message}`);
                    socket.write("message.type should be subscription_request, unsubscription_request or maneuver_recommendation");
            }
        } catch (exception) {
            console.log(`received invalid json data ${data}`);
            socket.write(`Error invalid json ${exception.toString()}`);
        }

    }

    handleSubscriptionRequest(message, socket) {
    	console.log("received subscription request");
    	socket.write(`{"type":"subscription_response","context":"subscriptions","origin":"injector","version":"0.1.0","timestamp":1549366070634,"source_uuid":"injector","message":{"result":"accept","request_id":14289383,"subscription_id":1},"signature":"TEMPLATE"}\n`);
    	        let notification = '{"type":"notify_add","context":"subscriptions","origin":"gdm","version":"1.2.0","source_uuid":"gdm","destination_uuid":"traffic_orchestrator_1970675314","timestamp":1557407399975,"message":{"subscription_id":1,"ru_description_list":[{"type":"ru_description","context":"general","origin":"self","source_uuid":"gdm","version":"1.2.0","message_id":"OBU11/OBU11/1557407399871","timestamp":1557407399975,"message":{"uuid":"OBU11","its_station_type":"passengerCar","connected":true,"position":{"latitude":486240452,"longitude":22396080},"position_type":"gnss_raw_rtk","heading":0,"speed":0,"acceleration":1,"yaw_rate":0,"raw_data":false,"size":{"length":0.0,"width":0.0,"height":0.0},"color":"0x000000","lane_position":0,"existence_probability":100,"accuracy":{"position_semi_major_confidence":5,"position_semi_minor_confidence":5,"position_semi_major_orientation":5,"heading":2,"speed":10,"size":{"length":1.0,"width":1.0,"height":1.0}}},"extra":[{"owner":"orange-labs","status":[]}],"signature":"signature"}]},"message_id":"gdm/1/1557407399975","signature":"signature"}';

        while (notification !== "") {
            const part = notification.substr(0, 1);
            notification = notification.substr(1);
            socket.write(part);
        }

        socket.write("\n");
        console.log("wrote notification in chunks");
    }

}

const tcp = new ToTCPServer();

tcp.start();
