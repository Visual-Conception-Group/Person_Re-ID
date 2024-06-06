import { Component, OnInit, SimpleChanges } from "@angular/core";
import { DomSanitizer } from "@angular/platform-browser";
import { ActivatedRoute, NavigationExtras, Router } from "@angular/router";
import { interval, Observable, Observer } from "rxjs";
import { VideoProcessService } from "../service/videoprocess.service";
import { ImageCroppedEvent, LoadedImage } from 'ngx-image-cropper';

declare var $: any;
declare var jQuery: any;
@Component({
    selector: "person-reid",
    templateUrl: "prid.component.html",
    styleUrls: ["./prid.component.css"]
})

export class PridComponent implements OnInit {
    imageChangedEvent: any = '';
    imageBase64String: any = ''
    frame_info = '';
    popup = false
    croppedImage: any = '';
    username: any = ""
    video_name: String = ""
    aux_video_names: String[] = []
    aux_video_paths: String[] = []
    video_duration: String = ""
    no_of_frames_extracted: String = ""
    video_path: String = ""
    list_of_seconds: any;
    total_frame_count: number = 0;
    selected_second: String = "";
    second_data: any = {
        "normal_frame_name": "",
        "normal_frame": "",
        "annotated_frame_name": "",
        "annotated_frame": "",
        "person_images": []

    };
    time_benchmark: any = {
        "time_to_get_images_for_prid" : "0",
        "time_to_extract_for_prid" : "0",
        "time_to_match" : "0",
        "time_to_read" : "0",
        "time_to_annotate" : "0"
    };
    displayStyle = "none";
    prid_result = new Array();
    containers = new Array();
    service_list = new Array();
    public constructor(private _video_service: VideoProcessService,
        private _activate_route: ActivatedRoute,
        private sanitizer: DomSanitizer,
        private route_service : Router) { }
    ngOnInit(): void {
        this._activate_route.queryParams.subscribe((data: any) => {
            // console.log(data)
            this.username = data["username"]
        })
        jQuery('#exampleModalCenter').modal({ 'backdrop': 'static', 'keyboard': false })
    }
    ngOnChanges(changes: SimpleChanges): void {
        // console.log(changes)

    }

    public get_video_info_v2() {
        let get_video_preview_service = this._video_service.get_video_preview_v2(this.username).subscribe((response: any) => {
            // console.log(response)
            this.video_name = response["video_name"]
            this.video_duration = response["video_duration"]
            this.no_of_frames_extracted = response["no_of_frame_extract"]
            this.video_path = this._video_service.httpurl + "/" + response["video_path"]
            this.aux_video_names = response["aux_videos"]
            this.aux_video_paths = response["aux_videos"].map((video: String) => this._video_service.httpurl + "/" + video)
        })

        if (!this.service_list.includes(get_video_preview_service))
            this.service_list.push(get_video_preview_service)
    }

    public get_frame_count_v2() {
        this.containers = new Array()
        // jQuery("#exampleModalCenter").modal('show');

        // let interval_status = interval(5000).subscribe(x => {
        //     this._video_service.get_status_api(this.username).subscribe((response: any) => {
        //         let index = -1;
        //         for (let i = 0; i < this.containers.length; i++) {
        //             if (this.containers[i]["status_title"] == response["title"]) {
        //                 index = i;
        //                 break
        //             }
        //         }
        //         if (index == -1) {
        //             for (let i = 0; i < this.containers.length; i++) {
        //                 this.containers[i]["status_value"] = true;
        //             }
        //             this.containers.push({
        //                 "status_title": response["title"],
        //                 "status_value": false
        //             })
        //         }

        //     });
        // })
        
        let get_frame_service = this._video_service.get_frame_count_v2(this.username).subscribe((response: any) => {
            // console.log(response)
            // jQuery("#exampleModalCenter").modal('hide');
            // console.log("Hello")
            // interval_status.unsubscribe()
            this.list_of_seconds = response["count"]
            this.total_frame_count = response["tot_count"]
            this.time_benchmark = {
                "time_to_read" : response["time_benchmark"]["time_read_video"],
                "time_to_annotate" : response["time_benchmark"]["time_annotate"]
            };
            this.containers = Array()
        })
        // if (!this.service_list.includes(interval_status))
        //     this.service_list.push(interval_status)
        if (!this.service_list.includes(get_frame_service))
            this.service_list.push(get_frame_service);
    }

    public get_second_info_v2() {
        this.second_data = {
            "normal_frame_name": "",
            "normal_frame": "",
            "annotated_frame_name": "",
            "annotated_frame": "",
            "person_images": [],
            "no_of_people_in_frame": 0

        };
        let get_second_info_api_service = this._video_service.get_second_info_api_v2(this.username, this.selected_second).subscribe((response: any) => {
            // console.log(response);
            this.second_data["normal_frame_name"] = response["normal_frame"]["image_name"]
            this.second_data["normal_frame"] = "data:image/png;base64," + response["normal_frame"]["image"]
            this.second_data["no_of_people_in_frame"] = response["object_count"]

        })
        if (!this.service_list.includes(get_second_info_api_service))
            this.service_list.push(get_second_info_api_service)
    }

    public start_reid_v2() {
        this.containers = new Array()
        // jQuery("#exampleModalCenter").modal('show');
        // let interval_status1 = interval(5000).subscribe(x => {
        //     this._video_service.get_status_api(this.username).subscribe((response: any) => {
        //         let index = -1;
        //         for (let i = 0; i < this.containers.length; i++) {
        //             if (this.containers[i]["status_title"] == response["title"]) {
        //                 index = i;
        //                 break
        //             }
        //         }
        //         if (index == -1) {
        //             for (let i = 0; i < this.containers.length; i++) {
        //                 this.containers[i]["status_value"] = true;
        //             }
        //             this.containers.push({
        //                 "status_title": response["title"],
        //                 "status_value": false
        //             })
        //         }



        //     });
        // })
        this.prid_result = new Array()
        const formData = new FormData();
        const file = this.DataURIToBlob(this.croppedImage)
        formData.append("query_image", file);
        formData.append("username", this.username);
        formData.append("frame_info", this.second_data["normal_frame_name"]);
        let prid_service = this._video_service.start_prid_api_v2(formData).subscribe((response: any) => {
            // console.log(response)
            for (let i = 0; i < response["prid_result"].length; i++) {
                // console.log(response["prid_result"][i])
                let objectURL = "data:image/jpeg;base64," + response["prid_result"][i]["image"]
                this.prid_result.push({
                    "frame": this.sanitizer.bypassSecurityTrustUrl(objectURL),
                    "frame_name": response["prid_result"][i]["image_name"]
                })
            }
            this.time_benchmark["time_to_get_images_for_prid"] = response["time_benchmark"]["time_to_get_images"],
            this.time_benchmark["time_to_match"] = response["time_benchmark"]["time_to_match"],
            this.time_benchmark["time_to_extract_for_prid"] = response["time_benchmark"]["time_to_extract"]
            // jQuery("#exampleModalCenter").modal('hide');
            // interval_status1.unsubscribe()
            this.containers = Array()
            // console.log("Done")
        })
        
        // if (!this.service_list.includes(interval_status1))
        //     this.service_list.push(interval_status1)
        if (!this.service_list.includes(prid_service))
            this.service_list.push(prid_service);
    }

    DataURIToBlob(dataURI: string) {
        const splitDataURI = dataURI.split(',')
        const byteString = splitDataURI[0].indexOf('base64') >= 0 ? atob(splitDataURI[1]) : decodeURI(splitDataURI[1])
        const mimeString = splitDataURI[0].split(':')[1].split(';')[0]

        const ia = new Uint8Array(byteString.length)
        for (let i = 0; i < byteString.length; i++)
            ia[i] = byteString.charCodeAt(i)

        return new Blob([ia], { type: mimeString })
    }
    public model_close() {
        jQuery("#exampleModalCenter").modal('hide');
    }

    imageCropped(event: ImageCroppedEvent) {
        this.croppedImage = event.base64;
    }
    imageLoaded(image: LoadedImage) {
        // show cropper
    }
    cropperReady() {
        // cropper ready
    }
    loadImageFailed() {
        // show message
    }

    public analysis_done() {
        this._video_service.analysis_done(this.username).subscribe((response: any) => {
            // console.log(response);
        });
    }
    public upload_again() {
        this._video_service.analysis_done(this.username).subscribe((response: any) => {
            // console.log(response);
        });
        let navigationExtras: NavigationExtras = {
            queryParams: {
                username: this.username,
            },skipLocationChange:true
        }
       this.route_service.navigate(["/videoprocess"],navigationExtras);
    }
}