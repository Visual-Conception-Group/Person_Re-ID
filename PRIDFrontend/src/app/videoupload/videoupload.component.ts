import { HttpEventType } from "@angular/common/http";
import {Component,OnInit} from "@angular/core";
import { VideoProcessService } from "../service/videoprocess.service";
import {ActivatedRoute, NavigationExtras, Router} from "@angular/router"
@Component({
    selector : "video-process",
    templateUrl : "videoupload.component.html",
    styleUrls : ["./videoupload.component.css"]
})
export class VideoUploadComponent implements OnInit{
    video_file :any;
    error_message : String = ""
    progress = 0;
    username : string = ""
    is_upload_complete : boolean = false;

    aux_video_files : any[] = [];
    aux_error_messages : String[] = [""]
    aux_progress = [0];
    is_aux_video_upload_complete : [boolean] = [false];

    public constructor(private __video_service : VideoProcessService,
        private _router : Router,
        private _activate_route : ActivatedRoute){

    }
    ngOnInit(): void {
        this._activate_route.queryParams.subscribe((data:any)=>{
            // console.log(data);
            this.username = data["username"]
        });
    }
    public changeInFileInput(event:any){
        this.video_file = event.target.files[0];
    }
    public changeInFileInputV2(event:any){
        this.aux_video_files = event.target.files;
    }
    public file_upload()
    {
        this.is_upload_complete = false;
        if(this.video_file){
            this.error_message = "Upload not yet completed....";
            const formData = new FormData();
            formData.append("video", this.video_file);
            formData.append("username" , this.username)
            this.__video_service.upload_video(formData).subscribe((event:any)=>{
                if (event.type == HttpEventType.UploadProgress) {
                    // console.log(Math.round(100 * (event.loaded / event.total)));
                    this.progress = Math.round(100 * (event.loaded / event.total));
                  }
                  if(event.type == HttpEventType.Response){
                    this.error_message = "Upload completed now.";
                    this.is_upload_complete = true;
                  }
            });    
        }else{
            this.error_message = "Please upload file"
        }
    }
    public go_to_video_process()
    {
        let navigationExtras: NavigationExtras = {
            queryParams: {
                username: this.username,
            },skipLocationChange:true
        }
        this._router.navigate(["/person-reid"],navigationExtras)
    }

    public file_upload_v2() {
        this.is_upload_complete = false;
        if(this.video_file){
            this.error_message = "Upload not yet completed....";
            const formData = new FormData();
            formData.append("video", this.video_file);
            
            for (var i = 0; i < this.aux_video_files.length; i++) { 
                formData.append("aux_video_" + Number(i+1), this.aux_video_files[i])
            }
            
            formData.append("username" , this.username)
            this.__video_service.upload_video_v2(formData).subscribe((event:any)=>{
                if (event.type == HttpEventType.UploadProgress) {
                    // console.log(Math.round(100 * (event.loaded / event.total)));
                    this.progress = Math.round(100 * (event.loaded / event.total));
                  }
                  if(event.type == HttpEventType.Response){
                    this.error_message = "Upload completed now.";
                    this.is_upload_complete = true;
                  }
            });    
        }else{
            this.error_message = "Please upload file"
        }
    }
}